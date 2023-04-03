import json
import os

import deepspeed
import torch
import numpy
from beartype import beartype
from beartype.typing import Optional, Tuple
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)

from chatllama.rlhf.config import ConfigActor
from chatllama.rlhf.model_list import (
    llama_models,
    hf_models_seq_2_seq,
    hf_models_causal_lm,
)
from chatllama.rlhf.utils import TrainingStats


class ActorModel(torch.nn.Module):
    """Actor model that generates the augmented prompt from the initial
    user_input. The aim is to train this model to generate better prompts.

    Attributes:
        model: The model from LLaMA to be used
        tokenizer: The LLaMA tokenizer
        config (ConfigActor): Configuration for the actor model

    Methods:
        load: Load the model from a path
        save: Save the model to a path
        forward: Compute the action logits for a given sequence.
        generate: Generate a sequence from a given prompt
    """

    def __init__(self, config: ConfigActor) -> None:
        super().__init__()
        # load the model

        if config.model in llama_models:
            # llama module might not be present when HF models are used
            from chatllama.llama_model import (
                load_model,
                setup_model_parallel,
                setup_model_deepspeed
            )  # noqa

            if config.deepspeed_enable:
                local_rank, world_size = setup_model_deepspeed()
            else:
                local_rank, world_size = setup_model_parallel()
            # use load_model_test for testing
            self.model, self.tokenizer = load_model(
                ckpt_dir=config.model_path,
                tokenizer_path=config.tokenizer_folder,
                local_rank=local_rank,
                world_size=world_size,
                froze_embeddings=config.froze_embeddings,
                use_fairscale=config.use_fairscale,
                max_batch_size=config.batch_size,
            )
        elif config.model in hf_models_seq_2_seq:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model,
                padding_side="left",
            )
            if self.tokenizer.eos_token is None:
                self.tokenizer.eos_token = "</s>"
                self.tokenizer.eos_token_id = 0
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                config.model,
            )
            self.model.to(config.device)
        elif config.model in hf_models_causal_lm:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model,
                padding_side="left",
            )
            # galactica tokenizer eos_token is None
            if self.tokenizer.eos_token is None:
                self.tokenizer.eos_token = "</s>"
                self.tokenizer.eos_token_id = 0
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model,
            )
            self.model.to(config.device)

        # save config
        self.config = config

    def parameters(self, **kwargs):
        """Return the parameters of the model

        Args:
            **kwargs:
        """
        return self.model.parameters()

    @beartype
    def forward(
        self, sequences: torch.Tensor, sequences_mask: torch.Tensor
    ) -> torch.Tensor:
        """Generate logits to have probability distribution over the vocabulary
            of the actions

        Args:
            sequences (torch.Tensor): Sequences of states and actions used to
                    compute token logits for the whole list of sequences
            attention_mask (torch.Tensor): Mask for the sequences attention

        Returns:
            logits (torch.Tensor): Logits for the actions taken
        """
        model_output = self.model.forward(
            sequences, attention_mask=sequences_mask
        )
        # need to return logits for the actions
        if self.config.model in hf_models_causal_lm:
            model_output = model_output.logits
        if self.config.debug:
            print("ActorModel.forward")
            print("model_output_logits shape", model_output.shape)
            print("model_output logits", model_output)
        return model_output

    @beartype
    @torch.no_grad()
    def generate(
        self, states: torch.Tensor, state_mask: torch.Tensor
    ) -> Tuple:
        """Generate actions and sequences=[states, actions] from state
            (i.e. input of the prompt generator model)

        Args:
            state (torch.Tensor): the input of the user
            state_mask (torch.Tensor): Mask for the state input (for padding)

        Returns:
            actions (torch.Tensor): Actions generated from the state
            sequences (torch.Tensor): Sequences generated from the
                state as [states, actions]
        """
        temperature = self.config.temperature
        # max sequence length for the actor (i.e. prompt + completion)
        # from config file - it depends by the model used
        max_sequence_length = self.config.max_sequence_length
        # max tokens generated by the actor (completion only) from config file
        max_tokens = self.config.max_tokens
        # temperature for the actor
        max_generation_possible = max_sequence_length - states.shape[1]
        # take the minimum between the maximum token that you want to generate
        # and the token that is possible to generate given the maximum sequence
        # supported
        max_completion = min(max_tokens, max_generation_possible)
        if max_completion <= 0:
            raise ValueError(
                "The maximum completion available is <= 0 the prompt is too "
                + "long w.r.t the model sequence length"
            )
        # the max_length is then the input length + the completion length
        max_length = states.shape[1] + max_completion
        # generate
        sequences = self.model.generate(
            input_ids=states,
            attention_mask=state_mask,
            temperature=temperature,
            max_length=max_length,
        )
        actions = sequences[:, states.shape[1] :]  # noqa E203
        if self.config.debug:
            print("ActorModel.generate")
            print("state", states)
            print("state shape", states.shape)
            print("sequence shape", sequences.shape)
            print("sequence", sequences)
            print("actions shape", actions.shape)
            print("actions", actions)
        return actions, sequences

    @beartype
    def load(self, path: Optional[str] = None) -> None:
        """Load the model from the path

        Args:
            path (str): Path to the model
        """
        if path is None:
            if self.config.model in hf_models_causal_lm:
                model_name = os.path.split(self.config.model)[-1]
            else:
                model_name = self.config.model
            path = os.path.join(
                self.config.checkpoint_folder, f"{model_name}.pt"
            )
            if os.path.exists(self.config.checkpoint_folder) is False:
                os.mkdir(self.config.checkpoint_folder)
                print(
                    f"Impossible to load the model: {path}"
                    f"The path doesn't exist."
                )
                return
        # load the model
        if os.path.exists(path) is False:
            print(
                f"Impossible to load the model: {path}"
                f"The path doesn't exist."
            )
            return
        model_dict = torch.load(path)
        self.model.load_state_dict(model_dict["model"])

    @beartype
    def save(self, path: Optional[str] = None) -> None:
        """Save the model to the path

        Args:
            path (Optional[str], optional): Path to store the model.
                Defaults to None.
        """
        if path is None:
            if self.config.model in hf_models_causal_lm:
                model_name = os.path.split(self.config.model)[-1]
            else:
                model_name = self.config.model
            path = os.path.join(
                self.config.checkpoint_folder, f"{model_name}.pt"
            )
            if os.path.exists(self.config.checkpoint_folder) is False:
                os.mkdir(self.config.checkpoint_folder)
        torch.save({"model": self.model.state_dict()}, path)


class ActorDataset(Dataset):
    """Dataset for the pretraining of the actor model
    read a json file with the following format:
    [
        {
            "user_input": "..."
            "completion": "..."
        },
        ...
    ]
    Where:
        user_input: the input of the user
        completion: the output of the user
    """

    def __init__(self, path: str) -> None:
        self.path = path
        with open(path, "r") as f:
            data = json.load(f)
            self.data = [d["user_input"] + " \t" + d["completion"] for d in data]
        self.len = len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(
        self,
    ):
        return self.len


class ActorTrainer:
    """Used to pre-train the actor model to generate better prompts.

    Args:
        config (ConfigActor): Configuration for the actor model

    Attributes:
        config (ConfigActor): Configuration for the actor model
        model (ActorModel): Actor model
        loss_function (torch.nn.CrossEntropyLoss): Loss function
        optimizer (torch.optim.Adam): Optimizer
        validation_flag (bool): Flag to indicate if the validation dataset
            is provided
        training_stats (TrainingStats): Training statistics

    Methods:
        train: Train the actor model
    """

    def __init__(self, config: ConfigActor) -> None:
        # load the model, optimizer, loss function and config
        self.config = config
        self.model = ActorModel(config)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.lr
        )

        # check checkpoint, datasets and other data
        if not os.path.exists(config.model_path):
            os.mkdir(config.model_path)
        self.validation_flag = False
        self.training_stats = TrainingStats()
        if config.validation_dataset_path is not None:
            self.validation_flag = True

        # create dataloaders
        self.train_dataset = ActorDataset(config.train_dataset_path)
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=config.batch_size
        )
        if self.validation_flag:
            self.eval_dataset = ActorDataset(config.validation_dataset_path)
            self.validation_dataloader = DataLoader(
                self.eval_dataset, batch_size=config.batch_size
            )

        # initialize deepspeed
        self.model_engine = None
        if config.deepspeed_enable is True:
            if config.deepspeed_config_path is None:
                raise ValueError(
                    "DeepSpeed config path is None, but deepspeed is enabled"
                )
            if os.path.exists(config.deepspeed_config_path) is False:
                raise ValueError(
                    f"DeepSpeed config path {config.deepspeed_config_path}"
                    f"does not exist"
                )
            (
                self.model_engine,
                self.optimizer,
                self.train_dataloader,
                _,
            ) = deepspeed.initialize(
                args=None,
                model=self.model,
                model_parameters=self.model.parameters(),
                training_data=self.train_dataset, # self.train_dataloader → self.train_dataset
                config=self.config.deepspeed_config_path,
            )

    def train(
        self,
    ) -> None:
        print("Start Actor Model Pretraining")
        # get config parameters
        batch_size = self.config.batch_size
        epochs = self.config.epochs
        device = self.config.device

        # compute the number of iterations
        n_iter = int(len(self.train_dataset) / batch_size)

        # traing loop
        for epoch in range(epochs):
            self.model.train()
            for i, input_output in enumerate(self.train_dataloader):
                with torch.no_grad():
                    input_output_tokenized = self.model.tokenizer(
                        input_output,
                        return_tensors="pt",
                        padding=True,
                    )
                    '''
                    input_output_tokenized
                    type: dist
                    key: input_ids, attention_mask, output_pos
                    ### input_ids ### type: tensor, size: [bs, max_len] (token_id)
                    ### attention_mask ### type: tensor, size: [bs, max_len] (0 or 1)
                    ### output_pos ### type: list, size: [bs] (pos_index)
                    '''
                    output_pos = input_output_tokenized["output_pos"]
                    training_output = input_output_tokenized["input_ids"][
                        :, 1:
                    ]
                    training_input = input_output_tokenized["input_ids"][
                        :, :-1
                    ]
                    attention_mask = input_output_tokenized["attention_mask"][
                        :, :-1
                    ]
                    training_output = training_output.to(device)
                    training_input = training_input.to(device)
                    attention_mask = attention_mask.to(device)

                # forward pass
                if self.config.deepspeed_enable:
                    est_output = self.model_engine(
                        training_input, attention_mask
                    )
                else:
                    est_output = self.model(training_input, attention_mask)

                loss = 0.
                for batch_i in range(len(output_pos)):
                    est_output_ = est_output[batch_i][output_pos[batch_i]:]
                    training_output_ = training_output[batch_i][output_pos[batch_i]:]
                    '''
                    print out natural language for debugging
                    '''
                    # training_token = training_output_.to("cpu").numpy().tolist()
                    # print("\ngt\n")
                    # print(self.model.tokenizer.decode(training_token))
                    # _, output_token = torch.max(est_output_, dim=1)
                    # output_token = output_token.to("cpu").numpy().tolist()
                    # print("\noutput\n")
                    # print(self.model.tokenizer.decode(output_token))
                    loss += self.loss_function(est_output_, training_output_)
                loss = loss / len(output_pos) # average loss on one batch
                self.training_stats.training_loss.append(loss.item())

                # backward pass
                if self.config.deepspeed_enable:
                    self.model_engine.backward(loss)
                    self.model_engine.step()
                else:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # print progress
                if i % self.config.iteration_per_print == 0:
                    print(
                        f"Epoch: {epoch+1}/{epochs}, "
                        f"Iteration: {i+1}/{n_iter}, "
                        f"Training Loss: {loss}"
                    )

                # save checkpoints
                steps_per_checkpoint = 2000 # setup here
                if i % steps_per_checkpoint == 0 and i != 0:
                    print("Saving the checkpoint...")
                    path = os.path.join(
                        self.config.checkpoint_folder,
                        self.config.model + "_epoch" + str(epoch) + "_step" + str(i) + ".pt"
                    )
                    self.model.save(path=path)

            if self.validation_flag:
                self.model.eval()
                for i, input_output in enumerate(self.validation_dataloader):
                    input_output_tokenized = self.model.tokenizer(
                        input_output, return_tensors="pt", padding=True
                    )
                    output_pos = input_output_tokenized["output_pos"]
                    validation_output = input_output_tokenized["input_ids"][
                        :, 1:
                    ]
                    validation_input = input_output_tokenized["input_ids"][
                        :, :-1
                    ]
                    attention_mask = input_output_tokenized["attention_mask"][
                        :, :-1
                    ]

                    # forward pass
                    est_output = self.model.forward(
                        validation_input, attention_mask
                    )

                    loss = 0.
                    for batch_i in range(len(output_pos)):
                        validation_output_ = validation_output[batch_i][output_pos[batch_i]:]
                        est_output_ = est_output[batch_i][output_pos[batch_i]:]
                        '''
                        print out natural language for debugging
                        '''
                        # validation_token = validation_output_.to("cpu").numpy().tolist()
                        # print("\ngt\n")
                        # print(self.model.tokenizer.decode(validation_token))
                        # _, output_token = torch.max(est_output_, dim=1)
                        # output_token = output_token.to("cpu").numpy().tolist()
                        # print("\noutput\n")
                        # print(self.model.tokenizer.decode(output_token))
                        loss += self.loss_function(est_output_, validation_output_)
                    loss = loss / len(output_pos)
                    self.training_stats.validation_loss.append(loss.item())
                    
                    # print progress
                    if i % self.config.iteration_per_print == 0:
                        print(
                            f"Epoch: {epoch+1}/{epochs}, "
                            f"Iteration: {i+1}/{n_iter}, "
                            f"Validation Loss: {loss}"
                        )

            print("One epoch ended, saving the model...")
            path = os.path.join(
                self.config.checkpoint_folder,
                self.config.model + "_epoch" + str(epoch) + "_final.pt"
            )
            self.model.save(path=path)

        print("Training Finished ")
