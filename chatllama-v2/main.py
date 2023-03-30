# Command:
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed main.py chatllama_configs/config_uie.yaml --type ACTOR

import argparse

from chatllama.rlhf.actor import ActorTrainer
from chatllama.rlhf.config import Config
from chatllama.rlhf.reward import RewardTrainer
from chatllama.rlhf.trainer import RLTrainer

import os

# Setup argument parser
parser = argparse.ArgumentParser(
    prog="main.py", description="RLHF Training of ChatBots"
)

parser.add_argument("configfile", help="Path to config.yaml file")
parser.add_argument(
    "-t",
    "--type",
    help=(
        "Specify the training type. RL: Training of the model using RL."
        "ACTOR: Training of the actor model. "
        "REWARD: Training of the reward model."
        "RL: The whole pipeline with the three training steps"
    ),
    default="ALL",
    choices=["ALL", "RL", "ACTOR", "REWARD"],
)
parser.add_argument(
    "-a", "--actor", help="Specify actor model by name", default=None
)
parser.add_argument(
    "-r", "--reward", help="Specify reward model by name", default=None
)
parser.add_argument(
    "-l", "--local_rank", type=str, help=""
)

# parse arguments
args = parser.parse_args()

# load config.yaml with all the project informations
config = Config(args.configfile)

# 
os.environ['RANK']=args.local_rank
os.environ['LOCAL_RANK']=args.local_rank

# overwrite config if specified differently
if args.actor is not None:
    config.actor.model = args.actor
if args.reward is not None:
    config.reward.model = args.reward

# perform the desired training
if args.type == "RL":
    rlhf_trainer = RLTrainer(config)
    rlhf_trainer.train()
elif args.type == "ACTOR":
    actor_trainer = ActorTrainer(config.actor)
    actor_trainer.train()
elif args.type == "REWARD":
    reward_trainer = RewardTrainer(config.reward)
    reward_trainer.train()
elif args.type == "ALL":
    reward_trainer = RewardTrainer(config.reward)
    reward_trainer.train()
    actor_trainer = ActorTrainer(config.actor)
    actor_trainer.train()
    rlhf_trainer = RLTrainer(config)
    rlhf_trainer.train()
