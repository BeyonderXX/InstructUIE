import json
import os
import random
from collections import deque, namedtuple

import deepspeed
import torch
from beartype import beartype
from beartype.typing import Deque, Tuple, List
from einops import rearrange
from torch.utils.data import Dataset, DataLoader

from chatllama.rlhf.actor import ActorModel
from chatllama.rlhf.config import ConfigReward, ConfigActor, Config
from chatllama.rlhf.reward import RewardModel, CriticModel
from chatllama.rlhf.utils import TrainingStats, ConversationLog


"""
train()
┌─────────────────────────────┐
│                             │◄─────────────────────────┐
│                             │                          │
│      ┌─────────────┐        │                          │
│      │ user input  │        │                          │ learn()
│      └─────┬───────┘        │             ┌────────────┴─────────────┐
│            │                │             │                          │
│            │                │             │       ┌────────┐         │
│            │                │             │   ┌───│ Update │──┐      │
│            │                │             │   │   └────▲───┘  │      │
│   ┌────────▼────────────┐   │             │   │        │      │      │
│   │  Actor (LLM Model)  │   │             │   │     ┌──┴───┐  │      │
│   └────────┬────────────┘   │             │   │     │ PPO  │  │      │
│            │                │             │   │     └▲────▲┘  │      │
│            │                │             │   │      │    │   │      │
│            │                │             │   │      │    │   │      │
│    ┌───────▼──────┐         │             │ ┌─▼──────┴┐ ┌─┴───▼──┐   │
│    │ Reward Model │         │             │ │  Actor  │ │ Critic │   │
│    └──────────────┘         │             │ └─────────┘ └────────┘   │
│                             │             │                          │
│                             │ x Episodes  └─────────────▲────────────┘
└───────────────┬─────────────┘                           │   x Epochs
                │ store N Examples per Timestep           │  
         ┌──────▼──────┐                                  │
         │             │                                  │
         │  Memories   ├──────────────────────────────────┘
         │             │ (update timesteps x N Examples)
         └─────────────┘
"""  # noqa W291


class ActorCritic(torch.nn.Module):
    """Actor Critic class stores both the actor and the critic models
    and it generates values and action for given sequences during the training
    of the actor.

    Attributes:
        actor (ActorModel): Actor model
        critic (CriticModel): Critic model
        debug (bool): enable prints for Debugging

    Methods:
        forward: given a sequence returns action logits and values (used
            to evaluate the actor during training)
        generate: given a sequence returns action, action logits, values
            sequences and sequences masks (used to generate new sequences
            during acting phase)
    """

    def __init__(
        self, actor_config: ConfigActor, critic_config: ConfigReward
    ) -> None:
        super().__init__()
        self.actor = ActorModel(actor_config)
        self.critic = CriticModel(critic_config)
        self.debug = actor_config.debug

    @beartype
    def forward(
        self,
        sequences: torch.Tensor,
        sequences_mask: torch.Tensor,
        action_len: int,
    ) -> Tuple:
        """Given the whole sequences, use the actor forward to get the logits
            for each token in the sequence and the critic forward to get the
            values for each generation step.

        Args:
            sequences (torch.Tensor): Sequences composed of [states, actions]
            sequence_mask (torch.Tensor): Mask for the sequences
            action_length (int): Length of the actions in the sequences

        Returns:
            action_logits (torch.Tensor): Logits for the actions in the
                sequences
            values (torch.Tensor): Values for the actions in the sequences
        """
        # use a single forward on the whole sequence
        # to get pi(y | x) and ignore predicted output
        actions_logits = self.actor.forward(sequences, sequences_mask)
        values = self.critic.forward(sequences, sequences_mask)

        # return only logits and values for the actions taken
        real_actions_logits = actions_logits[:, -action_len:, :]
        real_values = values[:, -action_len:]

        if self.debug:
            print("ActorCritic.forward")
            print("action_len", action_len)
            print("sequences.shape", sequences.shape)
            print("sequences", sequences)
            print("real_action_logits.shape", actions_logits.shape)
            print("real_action_logits", actions_logits)
            print("real_values.shape", values.shape)
            print("real_values", values)

        return (
            real_actions_logits,
            real_values,
        )

    @torch.no_grad()
    @beartype
    def generate(
        self, states: torch.Tensor, state_mask: torch.Tensor
    ) -> Tuple:
        """Generate actions, actions_logits, values and sequences from states

        Args:
            states (torch.Tensor): user inputs
            state_mask (torch.Tensor): Mask for the states of the environment

        Returns:
            actions (torch.Tensor): Actions generated from the states
            actions_logits (torch.Tensor): Logits for the actions generated
                from the states (i.e. pi(y | x))
            values (torch.Tensor): Values generated by the critic model
                for the actions generated by the actor (i.e. V(x))
            sequences (torch.Tensor): Sequences generated from the states
                as [states, actions]
        """
        # generate action sequence
        actions, sequence = self.actor.generate(states, state_mask)
        sequences_mask = sequence != self.actor.tokenizer.pad_token_id
        sequences_mask = sequences_mask.to(sequence.device).long().detach()
        action_len = actions.shape[1]

        # generate actions_logits and values
        actions_logits, values = self.forward(
            sequence, sequences_mask, action_len
        )
        if self.debug:
            print("ActorCritic.generate")
            print("actions shape", actions.shape)
            print("actions", actions)
            print("sequence shape", sequence.shape)
            print("sequence", sequence)
            print("actions_logits shape", actions_logits.shape)
            print("actions_logits", actions_logits)
            print("values shape", values.shape)
            print("values", values)

        return actions, actions_logits, values, sequence, sequences_mask


# structure to store the data for each experience
Memory = namedtuple(
    "Memory",
    [
        "states",
        "actions",
        "sequences",
        "values",
        "rewards",
        "actions_log_probs",
        "sequences_mask",
    ],
)


class ExperienceDataset(Dataset):
    """Dataset to train the actor-critic models"""

    def __init__(
        self,
        memories: Deque[Memory],
        device: torch.device,
    ) -> None:
        super().__init__()
        self.data = list(memories)
        self.device = device

    def __len__(
        self,
    ) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple:
        # return the idx-th memory element as a tuple of tensors on the device
        item = (
            self.data[idx].states.to(self.device),
            self.data[idx].actions.to(self.device),
            self.data[idx].sequences.to(self.device),
            self.data[idx].values.to(self.device),
            self.data[idx].rewards.to(self.device),
            self.data[idx].actions_log_probs.to(self.device),
            self.data[idx].sequences_mask.to(self.device),
        )
        return item


class ExamplesSampler:
    """Store the prompt to be sampled to generate the examples
    read a json file with the following format:
    [
        {
            "user_input" : "",
        } ,
        ...
    ]
    Where:
        user_input: is the input of the user or directly the input of the user
            with the memory preappended (i.e. user_input + memory)
    """

    def __init__(
        self,
        path: str,
    ) -> None:
        self.path = path
        with open(path, "r") as f:
            data = json.load(f)
        self.data = [d["user_input"] for d in data]

    def sample(self, n: int) -> List:
        """Sample n examples from the data

        Args:
            n (int): Number of examples to sample
        """
        return random.sample(self.data, n)


class RLTrainer:
    """Train the actor-critic model using RL

    Attributes:
        config (Config): Configuration of the trainer
        debug (bool): Debug mode
        actorcritic (ActorCritic): Actor-critic model
        actor_optim (torch.optim): Optimizer for the actor
        critic_optim (torch.optim): Optimizer for the critic
        reward (RewardModel): Reward model
        training_stats (TrainingStats): Class to store training stats
    Methods:
        train: the training loop that calls the learn function after generating
            the experiences.
        learn: Learn from a batch of experiences and update the actor and the
            critic model.
        load_checkpoint: Load the checkpoint of the actor-critic model
        save_checkpoint: Save the checkpoint of the actor-critic model
        generate_user_input: Generate the user input from the inputs
    """

    def __init__(
        self,
        config: Config,
    ) -> None:
        self.config = config
        self.debug = config.trainer.debug

        # initialize agent-critic
        self.actorcritic = ActorCritic(config.actor, config.critic)
        self.actor_optim = torch.optim.Adam(
            self.actorcritic.actor.parameters(), lr=config.trainer.actor_lr
        )
        self.critic_optim = torch.optim.Adam(
            self.actorcritic.critic.parameters(), lr=config.trainer.critic_lr
        )

        # initialize reward model
        self.reward = RewardModel(config.reward)

        # initialize class to store training stats
        self.training_stats = TrainingStats()
        self.conversation_log = ConversationLog()

        # initialize examples sampler
        self.example_sampler = ExamplesSampler(config.trainer.examples_path)

        # eps
        self.eps = 1e-8

        # make models directory
        if not os.path.exists("./models"):
            os.mkdir("./models")

        if not os.path.exists(self.config.trainer.checkpoint_folder):
            os.mkdir(self.config.trainer.checkpoint_folder)

    def save_checkpoint(
        self,
        current_episode: int,
    ) -> None:
        print(f"Saving checkpoint for episode {current_episode+1}..")
        file_name = "rltraining_" + str(current_episode) + ".pt"
        checkpoint_folder = self.config.trainer.checkpoint_folder
        if os.path.exists(checkpoint_folder) is False:
            os.mkdir(checkpoint_folder)
        path = checkpoint_folder + "/" + file_name
        torch.save(
            {
                "episode": current_episode,
                "actor_state_dict": self.actorcritic.actor.state_dict(),
                "critic_state_dict": self.actorcritic.critic.state_dict(),
                "actor_optim_state_dict": self.actor_optim.state_dict(),
                "critic_optim_state_dict": self.critic_optim.state_dict(),
                "training_stats": self.training_stats,
            },
            path,
        )

    def load_checkpoint(
        self,
    ) -> int:
        # get all the files name in the checkpoint folder and take the one
        # with the highest epoch
        checkpoint_folder = self.config.trainer.checkpoint_folder
        if os.path.exists(checkpoint_folder) is False:
            os.mkdir(checkpoint_folder)
            print(
                f"Checkpoint folder {checkpoint_folder} does not exist.\n"
                f"No checkpoint will be loaded."
            )
            return
        files = os.listdir(checkpoint_folder)
        episodes = [int(f.split("_")[1].split(".")[0]) for f in files]
        if len(episodes) == 0:
            return 0
        max_episode = max(episodes)
        print(f"Loading checkpoint for episode {max_episode+1}..")
        file_name = "rltraining_" + str(max_episode) + ".pt"
        path = checkpoint_folder + "/" + file_name
        checkpoint = torch.load(path)
        self.actorcritic.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actorcritic.critic.load_state_dict(
            checkpoint["critic_state_dict"]
        )
        self.actor_optim.load_state_dict(checkpoint["actor_optim_state_dict"])
        self.critic_optim.load_state_dict(
            checkpoint["critic_optim_state_dict"]
        )
        self.trainign_stats = checkpoint["training_stats"]
        self.actorcritic.actor.to(self.config.trainer.device)
        self.actorcritic.critic.to(self.config.trainer.device)
        return max_episode + 1  # return the next episode to train

    @beartype
    def learn(self, memories: Deque[Memory]) -> None:
        """Train the agent-critic model using RL:
        - for each batch of episodes, compute action logits and values
        - then compare action logits probs with memories one and values with
            rewards to compute the PPO loss and update the actor-critic model
        """
        print("Start to Learn...")

        # get parameters
        epochs = self.config.trainer.epochs
        actor_eps_clip = self.config.trainer.actor_eps_clip
        critic_eps_clip = self.config.trainer.critic_eps_clip
        beta_s = self.config.trainer.beta_s
        batch_size = self.config.trainer.batch_size
        device = self.config.trainer.device

        # create dataset from memories
        dataloader = DataLoader(
            ExperienceDataset(memories, device), batch_size=batch_size
        )

        # initialize deepspeed for actor
        if self.config.actor.deepspeed_enable:
            if self.config.actor.deepspeed_config_path is None:
                raise ValueError(
                    "DeepSpeed config path is None, but deepspeed is enabled"
                )
            if (
                os.path.exists(self.config.actor.deepspeed_config_path)
                is False
            ):
                raise ValueError(
                    f"DeepSpeed config path"
                    f"{self.config.actor.deepspeed_config_path}"
                    f"does not exist"
                )
            (
                actor_model_engine,
                actor_optimizer,
                dataloader,
                _,
            ) = deepspeed.initialize(
                args=None,
                model=self.actorcritic.actor,
                model_parameters=self.actorcritic.actor.parameters(),
                training_data=dataloader,
                config=self.config.actor.deepspeed_config_path,
            )
            self.actorcritic.actor = actor_model_engine

        # initialize deepspeed for critic
        if self.config.critic.deepspeed_enable:
            if self.config.critic.deepspeed_config_path is None:
                raise ValueError(
                    "DeepSpeed config path is None, but deepspeed is enabled"
                )
            if (
                os.path.exists(self.config.critic.deepspeed_config_path)
                is False
            ):
                raise ValueError(
                    f"DeepSpeed config path"
                    f"{self.config.critic.deepspeed_config_path}"
                    f"does not exist"
                )
            (
                critic_model_engine,
                critic_optimizer,
                _,
                _,
            ) = deepspeed.initialize(
                args=None,
                model=self.actorcritic.critic,
                model_parameters=self.actorcritic.critic.parameters(),
                config=self.config.critic.deepspeed_config_path,
            )
            self.actorcritic.critic = critic_model_engine

        # train agent-critic
        self.actorcritic.train()
        for epoch in range(epochs):
            for i, (
                states,
                old_actions,
                sequences,
                old_values,
                rewards,
                old_actions_log_probs,
                sequences_mask,
            ) in enumerate(dataloader):

                # print
                print(
                    "Epoch",
                    epoch + 1,
                    "of",
                    epochs,
                    "Data",
                    i + 1,
                    "of",
                    int(len(dataloader) / batch_size),
                )

                if self.debug:
                    print("RLTrainer.learn()")
                    print("memory states shapes are: ")
                    print("states shape", states.shape)
                    print("old_actions shape", old_actions.shape)
                    print("sequences shape", sequences.shape)
                    print("old_values shape", old_values.shape)
                    print("rewards shape", rewards.shape)
                    print(
                        "old_actions_log_probs shape",
                        old_actions_log_probs.shape,
                    )
                # reshaping rewards to match [b, s] shape
                rewards = rearrange(rewards, "b -> b 1")

                # get actions len
                actions_len = old_actions.shape[-1]

                # get actor critic forward
                actions_logits, values = self.actorcritic.forward(
                    sequences, sequences_mask, actions_len
                )

                # get action log prob
                actions_prob = (
                    torch.softmax(actions_logits, dim=-1).max(dim=-1).values
                )
                actions_log_prob = torch.log(actions_prob + self.eps)

                # compute entropy
                entropies = (actions_prob * actions_log_prob).sum(dim=-1)

                # compute KL divergence
                kl_div_loss = (
                    (actions_prob * (old_actions_log_probs - actions_log_prob))
                    .sum(dim=-1)
                    .mean()
                )

                # compute PPO Loss -- Whan dimensions are different
                # (especially the values and the probs are
                #  multiplied directly with the reward)
                ratios = (actions_log_prob - old_actions_log_probs).exp()
                advantages = rewards - old_values
                # normalize advantages
                advantages = (advantages - advantages.mean(dim=-1)) / (
                    advantages.std() + self.eps
                )
                surr1 = advantages * ratios
                surr2 = (
                    torch.clamp(ratios, 1 - actor_eps_clip, 1 + actor_eps_clip)
                    * advantages
                )
                policy_loss = -torch.min(surr1, surr2) - beta_s * entropies
                policy_loss = policy_loss.mean()
                loss = policy_loss + kl_div_loss
                # check if loss item is nan
                if torch.isnan(loss):
                    raise ValueError("Loss is nan")
                print("loss", loss.item())

                if self.debug:
                    print("values", values)
                    print("old_values", old_values)
                    print("rewards", rewards)
                    print("ratios", ratios)
                    print("advantages", advantages)
                    print("entropies", entropies)

                # update actor with loss
                if self.config.actor.deepspeed_enable:
                    actor_model_engine.backward(loss)
                    actor_model_engine.step()
                else:
                    self.actor_optim.zero_grad()
                    loss.backward()
                    self.actor_optim.step()

                torch.cuda.synchronize(device)

                # compute value loss
                value_loss_clipped = old_values + (values - old_values).clamp(
                    -critic_eps_clip, critic_eps_clip
                )
                value_loss1 = (value_loss_clipped - rewards) ** 2
                value_loss2 = (values - rewards) ** 2
                value_loss = torch.max(value_loss1, value_loss2).mean()
                if torch.isnan(value_loss):
                    raise ValueError("Value loss is nan")
                print("value_loss", value_loss.item())

                # upate critic with loss
                if self.config.critic.deepspeed_enable:
                    critic_model_engine.backward(value_loss)
                    critic_model_engine.step()
                else:
                    self.critic_optim.zero_grad()
                    value_loss.backward()
                    self.critic_optim.step()

                self.training_stats.training_loss.append(
                    loss.detach().cpu().item()
                )
                self.training_stats.value_loss.append(
                    value_loss.detach().cpu().item()
                )

        self.actorcritic.eval()
        print("End Learning")

    def train(
        self,
    ) -> None:
        # initialize settings
        num_episodes = self.config.trainer.num_episodes
        max_timesteps = self.config.trainer.max_timesteps
        num_examples = self.config.trainer.num_examples
        update_timesteps = self.config.trainer.update_timesteps
        batch_size = self.config.trainer.batch_size
        update_checkpoint = self.config.trainer.update_checkpoint
        device = self.config.trainer.device

        print("Start RL Training")
        # check dimensions consistency
        # at each time step num_examples memories are generated
        number_of_memories_per_learn_iteration = (
            num_examples * update_timesteps
        )
        # the number of memories must be a multiple of the batch size
        assert (
            number_of_memories_per_learn_iteration % batch_size == 0
        ), "The number of memories must be a multiple of the batch size"
        # the total number of timesteps is
        total_number_of_timesteps = num_episodes * max_timesteps
        # the update_timesteps must be a multiple
        #  of the total number of timesteps
        assert total_number_of_timesteps % update_timesteps == 0, (
            "The number of timesteps (num_episodes*max_timesteps)"
            "must be a multiple of the update_timesteps"
        )

        # initialize memories
        memories = deque([])

        # loop over episodes and timesteps
        current_time = 0
        checkpoint_counter = 0
        current_episode = self.load_checkpoint()
        current_learn_counter = 0

        self.actorcritic.eval()
        for eps in range(current_episode, num_episodes):
            for timestep in range(max_timesteps):

                print(
                    f"Episode: {eps + 1} of {num_episodes}, "
                    f"Timestep: {timestep + 1} of {max_timesteps}",
                )

                # counter used to count timesteps into memory
                current_time += 1

                # sample num_examples examples from  example dataset
                inputs = self.example_sampler.sample(num_examples)

                # tokenize examples
                tokenized_inputs = self.actorcritic.actor.tokenizer(
                    inputs, padding=True, return_tensors="pt"
                )
                if self.debug:
                    print("RLTrainer.train()")
                    print("tokenized inputs", tokenized_inputs)
                # states are [batch_size, seq_len_of_states]
                states = tokenized_inputs["input_ids"].to(device)
                states_mask = tokenized_inputs["attention_mask"].to(device)

                # generate prompts
                # actions --> output produced by the actor head in response
                #  of the state(input) [batch_size, len_of_actions]
                # actions_logits --> logits of the actions
                # [batch_size, len_of_actions, vocab_size]
                # values --> output produced by the critic for each action
                # [batch_size, len_of_actions]
                # sequence --> (state, actions)
                # [batch_size, len_of_actions + seq_len_of_states] =
                # [batch_size, seq_len]
                (
                    actions,
                    actions_logits,
                    values,
                    sequences,
                    sequences_mask,
                ) = self.actorcritic.generate(states, states_mask)

                # from action logits to action log probs
                action_prob = (
                    torch.softmax(actions_logits, dim=-1).max(dim=-1).values
                )
                actions_log_probs = torch.log(action_prob + self.eps)

                completions = [
                    self.actorcritic.actor.tokenizer.decode(action)
                    for i, action in enumerate(actions)
                ]
                if self.debug:
                    print("RLTrainer.train()")
                    print("completions:")
                    for i, completion in enumerate(completions):
                        print(i, completion)
                        print("")

                # compute reward for the completion
                # the reward must take into account the answer quality wrt to
                # the initial request given
                # and must be tokenized again
                task_responses = []
                for input, completion in zip(inputs, completions):
                    task_response = input + "\n" + completion
                    task_responses.append(task_response)
                if self.debug:
                    print("RLTrainer.train()")
                    print("task_responses:")
                    for i, task_response in enumerate(task_responses):
                        print(i, task_response)
                        print("")
                tokenized_responses = self.reward.tokenizer(
                    task_responses, padding=True, return_tensors="pt"
                )
                rewards = self.reward.get_reward(
                    tokenized_responses["input_ids"].to(device),
                    tokenized_responses["attention_mask"].to(device),
                )

                # store memories of the episode / timestep
                for i in range(states.shape[0]):
                    memories.append(
                        Memory(
                            *map(
                                lambda x: x.detach().cpu(),
                                (
                                    states[i, :],
                                    actions[i, :],
                                    sequences[i, :],
                                    values[i, :],
                                    rewards[i],
                                    actions_log_probs[i, :],
                                    sequences_mask[i, :],
                                ),
                            )
                        )
                    )

                # log the memories in the conversation log
                for i in range(states.shape[0]):
                    self.conversation_log.add_conversation(
                        inputs[i],
                        completions[i],
                        rewards[i].detach().cpu().item(),
                        current_learn_counter,
                    )

                # learn from memories
                print(
                    f"Learning counter: {current_time} of {update_timesteps}"
                )
                if (current_time % update_timesteps == 0) and (
                    current_time != 0
                ):
                    checkpoint_counter += 1
                    self.conversation_log.show(current_learn_counter)
                    self.learn(memories)
                    memories.clear()
                    current_time = 0
                    current_learn_counter += 1

                if (checkpoint_counter % update_checkpoint == 0) and (
                    checkpoint_counter != 0
                ):
                    self.save_checkpoint(eps)
                    checkpoint_counter = 0

        self.actorcritic.critic.save()
        self.actorcritic.actor.save()
        print("End RL Training")
