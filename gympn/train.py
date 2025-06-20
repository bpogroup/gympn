#!/usr/bin/env python
"""Entry point for all training runs."""

import argparse
import datetime
import json
import subprocess
import webbrowser
import time
import os
import torch
from gympn.environment import AEPN_Env
from gympn.networks import HeteroActor, HeteroCritic
from gympn.agents import PGAgent, PPOAgent


#train = True

def make_parser():
    """Return the command line argument parser for this script."""
    parser = argparse.ArgumentParser(description="Train a new model",
                                     fromfile_prefix_chars='@')

    env = parser.add_argument_group('environment', 'environment type')
    env.add_argument('--environment',
                     choices=['ActionEvolutionPetriNetEnv'],
                     default='ActionEvolutionPetriNetEnv',
                     help='training environment')
    env.add_argument('--env_seed',
                     type=lambda x: int(x) if x.lower() != 'none' else None,
                     default=None,
                     help='seed for the environment')

    alg = parser.add_argument_group('algorithm', 'algorithm parameters')
    alg.add_argument('--algorithm',
                     choices=['ppo-clip', 'ppo-penalty', 'pg'],
                     default='ppo-clip',
                     help='training algorithm')
    alg.add_argument('--gam',
                     type=float,
                     default=1,
                     help='discount rate')
    alg.add_argument('--lam',
                     type=float,
                     default=0.99,
                     help='generalized advantage parameter')
    alg.add_argument('--eps',
                     type=float,
                     default=0.2,
                     help='clip ratio for clipped PPO')
    alg.add_argument('--c',
                     type=float,
                     default=0.2,
                     help='KLD weight for penalty PPO')
    alg.add_argument('--ent_bonus',
                     type=float,
                     default=0.0,
                     help='bonus factor for sampled policy entropy')
    alg.add_argument('--agent_seed',
                     type=lambda x: int(x) if x.lower() != 'none' else None,
                     default=None,
                     help='seed for the agent')

    policy = parser.add_argument_group('policy model')
    policy.add_argument('--policy_model',
                        choices=['gnn'],
                        default='gnn',
                        help='policy network type')
    policy.add_argument('--policy_kwargs',
                        type=json.loads,
                        default={"hidden_layers": [64]},
                        help='arguments to policy model constructor, passed through json.loads')
    policy.add_argument('--policy_lr',
                        type=float,
                        default=3e-3,#3e-4,
                        help='policy model learning rate')
    policy.add_argument('--policy_updates',
                        type=int,
                        default=10, #10
                        help='policy model updates per epoch')
    policy.add_argument('--policy_kld_limit',
                        type=float,
                        default=1, #0.01
                        help='KL divergence limit used for early stopping')
    policy.add_argument('--policy_weights',
                        type=str,
                        default="",#"policy-500.h5",
                        help='filename for initial policy weights')
    policy.add_argument('--policy_network',
                        type=str,
                        default="",#"policy-500.pth",
                        help='filename for initial policy weights')
    policy.add_argument('--score',
                        type = lambda x: str(x).lower() == 'true',
                        default = False,
                        help = 'have multi objective training')
    policy.add_argument('--score_weight',
                        type = float,
                        default=1e-3,
                        help='weight gradients of l2 loss')

    value = parser.add_argument_group('value model')
    value.add_argument('--value_model',
                       choices=['none', 'gnn'],
                       default='gnn',
                       help='value network type')
    value.add_argument('--value_kwargs',
                       type=json.loads,
                       default={"hidden_layers": [64]},
                       help='arguments to value model constructor, passed through json.loads')
    value.add_argument('--value_lr',
                       type=float,
                       default=3e-3,
                       help='the value model learning rate')
    value.add_argument('--value_updates',
                       type=int,
                       default=40, #40
                       help='value model updates per epoch')
    value.add_argument('--value_weights',
                       type=str,
                       default="",
                       help='filename for initial value weights')

    train = parser.add_argument_group('training')
    train.add_argument('--episodes',
                       type=int,
                       default=20, #100
                       help='number of episodes per epoch')
    train.add_argument('--epochs',
                       type=int,
                       default=20, #2500
                       help='number of epochs')
    train.add_argument('--max_episode_length',
                       type=lambda x: int(x) if x.lower() != 'none' else None,
                       default=None, #500
                       help='max number of interactions per episode')
    train.add_argument('--batch_size',
                       type=lambda x: int(x) if x.lower() != 'none' else None,
                       default=64,
                       help='size of batches in training')
    train.add_argument('--sort_states',
                       type=lambda x: str(x).lower() == 'true',
                       default=False,
                       help='whether to sort the states before batching')
    train.add_argument('--use_gpu',
                       type=lambda x: str(x).lower() == 'true',
                       default=False,
                       help='whether to use a GPU if available')
    train.add_argument('--load_policy_network',
                       type=bool,
                       default=False,
                       help='wether to load a previously trained policy as starting point for this run')
    train.add_argument('--test_in_train',
                       type=lambda x: str(x).lower() == 'true',
                       default=True,
                       help='whether to test the agent during training')
    train.add_argument('--test_freq',
                       type=int,
                       default=1,
                       help='frequency (in epochs) to run testing during training')
    train.add_argument('--verbose',
                       type=int,
                       default=0,
                       help='how much information to print')

    save = parser.add_argument_group('saving')
    save.add_argument('--name',
                       type=str,
                       default='run',
                       help='name of training run')
    save.add_argument('--datetag',
                       type=lambda x: str(x).lower() == 'true',
                       default=True,
                       help='whether to append current time to run name')
    save.add_argument('--logdir',
                       type=str,
                       default='data/train',
                       help='base directory for training runs')
    save.add_argument('--save_freq',
                       type=int,
                       default=1,
                       help='how often to save the models (only if test_in_train==False)')
    save.add_argument('--open_tensorboard',
                      type=lambda x: str(x).lower() == 'true',
                      default=False,
                      help='whether to open tensorboard for this run')
    return parser



def make_env(args, aepn = None):
    """Return the training environment for this run."""
    if args.environment == 'ActionEvolutionPetriNetEnv':
        env = AEPN_Env(aepn)
    else:
        raise Exception("Unknown environment! Are you sure it is spelled correctly?")
    #env.seed(args.env_seed)
    env.reset()
    return env


def make_policy_network(args, metadata=None):
    """Return the policy network for this run."""
    if args.environment == 'ActionEvolutionPetriNetEnv':
        if args.load_policy_network:
            policy_network = torch.load(os.path.join(os.getcwd(), args.logdir, args.name, args.policy_network))
        elif args.environment == 'ActionEvolutionPetriNetEnv':
            policy_network = HeteroActor(
                input_size=args.policy_kwargs.get("input_size", -1),
                hidden_size=args.policy_kwargs.get("hidden_size", 256),
                output_size=args.policy_kwargs.get("output_size", 64),
                num_heads=args.policy_kwargs.get("num_heads", 1),
                metadata=metadata
            )
        else:
            raise Exception("No policy network to load!")
    else:
        raise Exception("Unknown environment! Are you sure it is spelled correctly?")
    return policy_network


def make_value_network(args, metadata=None):
    """Return the value network for this run."""
    if args.value_model == 'none':
        value_network = None
    elif args.environment == 'ActionEvolutionPetriNetEnv':
        value_network = HeteroCritic(
            input_size=args.value_kwargs.get("input_size", -1),
            hidden_size=args.value_kwargs.get("hidden_size", 256),
            output_size=args.value_kwargs.get("output_size", 64),
            num_heads=args.value_kwargs.get("num_heads", 1),
            metadata=metadata
        )
    else:
        raise Exception("Unknown environment! Are you sure it is spelled correctly?")
    if args.value_weights != "":
        value_network.load_weights(os.path.join(os.getcwd(), args.logdir, args.name, args.value_weights))
    return value_network


def make_agent(args, metadata=None):
    """Return the agent for this run."""
    policy_network = make_policy_network(args, metadata=metadata)
    value_network = make_value_network(args, metadata=metadata)

    if args.algorithm == 'pg':
        agent = PGAgent(policy_network=policy_network,policy_lr=args.policy_lr, policy_updates=args.policy_updates,
                        value_network=value_network, value_lr=args.value_lr, value_updates=args.value_updates,
                        gam=args.gam, lam=args.lam, kld_limit=args.policy_kld_limit, ent_bonus=args.ent_bonus)
    elif args.algorithm == 'ppo-clip':
        agent = PPOAgent(policy_network=policy_network, method='clip', eps=args.eps,
                         policy_lr=args.policy_lr, policy_updates=args.policy_updates,
                         value_network=value_network, value_lr=args.value_lr, value_updates=args.value_updates,
                         gam=args.gam, lam=args.lam, kld_limit=args.policy_kld_limit, ent_bonus=args.ent_bonus)
    elif args.algorithm == 'ppo-penalty':
        agent = PPOAgent(policy_network=policy_network, method='penalty', c=args.c,
                         policy_lr=args.policy_lr, policy_updates=args.policy_updates,
                         value_network=value_network, value_lr=args.value_lr, value_updates=args.value_updates,
                         gam=args.gam, lam=args.lam, kld_limit=args.policy_kld_limit, ent_bonus=args.ent_bonus)
    else:
        raise Exception("Unknown algorithm! Are you sure it is spelled correctly?")
    return agent


def make_logdir(args):
    """Return the directory name for this run."""
    run_name = args.name
    if args.datetag:
        time_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        run_name = time_string + '_' + run_name
    logdir = os.path.join(args.logdir, run_name)
    #make dir if it does not exist already
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    with open(os.path.join(logdir, 'args.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write('--' + arg + '\n')
            if isinstance(value, dict):
                f.write(json.dumps(value) + "\n")
            else:
                f.write(str(value) + '\n')
    return logdir


def launch_tensorboard(logdir, port=6006, wait_time=5, reload_interval=30):
    """
    Launch TensorBoard and open it in the default web browser.

    Parameters
    ----------
    logdir : str
        Path to the directory containing TensorBoard logs.
    port : int, optional
        Port to run TensorBoard on (default is 6006).
    wait_time : int, optional
        Time to wait (in seconds) for TensorBoard to start before opening the browser.
    reload_interval : int, optional
        Interval (in seconds) for TensorBoard to check for new data (default is 30).

    Returns
    -------
    tensorboard_process : subprocess.Popen
        The TensorBoard process running in the background.
    """
    # Start TensorBoard as a subprocess
    tensorboard_process = subprocess.Popen(
        [
            "tensorboard",
            "--logdir", logdir,
            "--port", str(port),
            "--reload_interval", str(reload_interval)
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Wait for TensorBoard to start
    time.sleep(wait_time)

    # Open TensorBoard in the default web browser
    webbrowser.open(f"http://localhost:{port}")

    return tensorboard_process