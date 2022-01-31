import argparse
import gym
import numpy as np
import os
import wandb
import utils
from custom_callbacks import CustomCallbacks
from shutil import copyfile
from gym_socialgame.envs.activity_agent import (JSONFileAutomator)

import ray
import ray.rllib.agents.ppo as ray_ppo

from gym_socialgame.envs.socialgame_env import (SocialGameEnvRLLib)

def get_agent(args):
    """
    Purpose: Import the algorithm and policy to create an agent. 
    Returns: Agent
    Exceptions: Malformed args to create an agent. 
    """
    #### Algorithm: PPO ####
    if args.algo == "ppo":
        # Modify the default configs for PPO
        config = ray_ppo.DEFAULT_CONFIG.copy()
        config["framework"] = "torch"
        config["train_batch_size"] = 256
        config["sgd_minibatch_size"] = 16
        config["lr"] = 0.001
        config["clip_param"] = 0.3
        # config["num_gpus"] =  1 # this may throw an error
        config["num_workers"] = 1
        config["env_config"] = vars(args)
        config["env"] = environments[args.gym_env]
        # obs_dim = np.sum([args.energy_in_state, args.price_in_state])
        obs_dim = 10
            
        out_path = os.path.join(args.log_path, "bulk_data.h5")
        callbacks = CustomCallbacks(log_path=out_path, save_interval=args.bulk_log_interval, obs_dim=obs_dim)
        config["callbacks"] = lambda: callbacks
        logger_creator = utils.custom_logger_creator(args.log_path)

        callbacks.save()
        print("Saved first callback")

        if args.wandb:
            wandb.save(out_path)

        return ray_ppo.PPOTrainer(
            config = config, 
            env = environments[args.gym_env],
            logger_creator = logger_creator
        )

    # Add more algorithms here. 

def train(agent, args):
    """
    Purpose: Train agent in environment. 
    """
    library = args.library
    algo = args.algo
    env = args.gym_env
    num_steps = args.num_steps

    if library == "rllib":
        print("Initializing Ray")
        # ray.init()
        print("Ray Initialized")

    ## Beginning Training ##
    print("Beginning training.")
    to_log = ["episode_reward_mean"]
    training_steps = 0
    while training_steps < num_steps:
        result = agent.train()
        training_steps = result["timesteps_total"]
        log = {name: result[name] for name in to_log}
        print(log)

    # callbacks.save() TODO: Implement callbacks
######################################
#### Arguments and Configurations ####
######################################

# Add environments here to be included when configuring an agent
environments = {
    "socialgame": SocialGameEnvRLLib,
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--library",
    help = "What RL Library backend is in use",
    type = str,
    default = "rllib",
    choices = ["rllib", "tune"]
)
# Algorithm Arguments
parser.add_argument(
    "--algo",
    help="RL Algorithm",
    type=str,
    default="ppo",
    choices=["sac", "ppo", "maml", "uc_bandit"]
)
parser.add_argument(
    "--action_space_string",
    help="action space for algorithm",
    default="continuous",
)
parser.add_argument(
    "--policy_type",
    help="Type of Policy (e.g. MLP, LSTM) for algo",
    default="mlp",
    choices=["mlp", "lstm"],
)
parser.add_argument(
    "--reward_function",
    help="reward function to test",
    type=str,
    default="log_cost_regularized",
    choices=["scaled_cost_distance", "log_cost_regularized", "log_cost", "scd", "lcr", "lc", "market_solving", "profit_maximizing"],
)
# Environment Arguments
parser.add_argument(
    "--gym_env", 
    help="Which Gym Environment you wihs to use",
    type=str,
    choices=["socialgame"],
    default="socialgame"
)
parser.add_argument(
    "--env_id",
    help="Environment ID for Gym Environment",
    type=str,
    choices=["v0", "monthly"],
    default="v0",
)
parser.add_argument(
    "--response_type_string",
    help="Player response function (l = linear, t = threshold_exponential, s = sinusoidal",
    type=str,
    default="l",
    choices=["l", "t", "s"],
)
# Experiment Arguments
parser.add_argument(
    "--exp_name",
    help="experiment_name",
    type=str,
    default="experiment"
)
parser.add_argument(
    "--num_steps",
    help="Number of timesteps to train algo",
    type=int,
    default=50000,
)
parser.add_argument(
    "--energy_in_state",
    help="Whether to include energy in state (default = F)",
    action="store_true"
)
parser.add_argument(
    "--price_in_state",
    help="Whether to include price in state (default = F)",
    action="store_false"
)
parser.add_argument(
    "--batch_size",
    help="Batch Size for sampling from replay buffer",
    type=int,
    default=5,
    choices=[i for i in range(1, 30)],
)
parser.add_argument(
    "--one_day",
    help="Specific Day of the year to Train on (default = 15, train on day 15)",
    type=int,
    default=15,
    choices=[i for i in range(365)],
)
parser.add_argument(
    "--manual_tou_magnitude",
    help="Magnitude of the TOU during hours 5,6,7. Sets price in normal hours to 0.103.",
    type=float,
    default=.4
)
parser.add_argument(
    "--pricing_type",
    help="time of use or real time pricing",
    type=str,
    choices=["TOU", "RTP"],
    default="TOU",
)
parser.add_argument(
    "--number_of_participants",
    help="Number of players ([1, 20]) in social game",
    type=int,
    default=10,
    choices=[i for i in range(1, 21)],
)
parser.add_argument(
    "--learning_rate",
    help="learning rate of the the agent",
    type=float,
    default=3e-4,
)
parser.add_argument(
    "--new_agents",
    help="Whether to create new agents and store in file.",
    action="store_true"
)
parser.add_argument(
    "--num_demand_units",
    help="Number of demand units to use.",
    type=int
)
parser.add_argument(
    "--num_activities",
    help="Number of activities to use.",
    type=int
)
parser.add_argument(
    "--num_activity_consumers",
    help="Number of activity consumers to use.",
    type=int
)
parser.add_argument(
    "--keep_json",
    help="Whether to save the json file when running the activity agent.",
    action="store_true"
)
# Logging Arguments
parser.add_argument(
    "-w",
    "--wandb",
    help="Whether to upload results to wandb. must have wandb key.",
    action="store_true"
)
parser.add_argument(
    # "--base_log_dir",
    "--log_path",
    help="Base directory for tensorboard logs",
    type=str,
    default="./logs/"
)
parser.add_argument(
    "--bulk_log_interval",
    help="Interval at which to save bulk log information",
    type=int,
    default=10000
)
parser.add_argument(
    "--bin_observation_space",
    help= "Whether to bin the observations.",
    action="store_true"
)
parser.add_argument(
    "--smirl_weight",
    help="Whether to run with SMiRL. When using SMiRL you must specify a weight.",
    type = float,
    default=None,
)
parser.add_argument(
    "--circ_buffer_size",
    help="Size of circular smirl buffer to use. Will use an unlimited size buffer in None",
    type = float,
    default=None,
)
# Machine Arguments
parser.add_argument(
    "-l",
    "--local_mode",
    help="Init Ray in local mode for easier debugging.",
    action="store_true"
)

#
# Call get_agent and train to recieve the agent and then train it. 
#
if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running with following options: {args}")

    # ray.init(local_mode=args.local_mode)
    # ray.init()
    # TODO

    # Uploading logs to wandb
    if args.wandb:
        if args.new_agents:
            size_props = JSONFileAutomator.edit_file(
                                                        reset_param = True, 
                                                        num_demand_units = args.num_demand_units,
                                                        num_activities = args.num_activities,
                                                        num_activity_consumers = args.num_demand_units,
                                                    )
        else:
            size_props = JSONFileAutomator.read_size_props()
        
        group_name = '{0} | {1} | {2}'.format(size_props[0], size_props[1], size_props[2])
        wandb.init(project="activity_agent_grouped", entity="joshlor", group=group_name) 

        if args.keep_json:
            folder_location = "gym-socialgame/gym_socialgame/envs/activity_environments/{file_name}"
            copyfile(folder_location.format(file_name = "activity_env.json"), folder_location.format(file_name = (wandb.run.name + ".json")))
        
        wandb.tensorboard.patch(root_logdir=args.log_path) # patching the logdir directly seems to work
        wandb.config.update(args)

    # Get Agent
    agent = get_agent(args)
    # TODO: Implement
    print("Agent initialied.")

    # Training
    print(f'Beginning Testing! Logs are being saved somewhere')
    # TODO: Implement
    train(agent, args)


