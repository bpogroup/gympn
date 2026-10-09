#!/usr/bin/env python
"""Entry point for all training runs."""

import argparse
import datetime
import json
import subprocess
import sys
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
    alg.add_argument('--vf_coeff',
                     type=float,
                     default=0.5,
                     help='value function loss coefficient in total loss')
    alg.add_argument('--agent_seed',
                     type=lambda x: int(x) if x.lower() != 'none' else None,
                     default=None,
                     help='seed for the agent')
    alg.add_argument('--nfgae',
                     type=lambda x: str(x).lower() == 'true',
                     default=False,
                     help='net-factored GAE (suite/paper/NFGAE_THEORY.md): credit each decision '
                          'only with the rewards of its own net component, on that component\'s '
                          'own decision clock, with a per-component critic. Needs '
                          'allow_postpone=False or postpone_scope=\'component\'. With a single '
                          'component it is exactly SMDP-GAE PPO.')
    alg.add_argument('--beta',
                     type=float,
                     default=0.0,
                     help='SMDP time-discount rate: the continuation after a decision is '
                          'discounted by exp(-beta * tau), tau = elapsed time to the next '
                          'decision. Used with --smdp_discount and --nfgae.')
    alg.add_argument('--smdp_discount',
                     type=lambda x: str(x).lower() == 'true',
                     default=False,
                     help='replace the constant-gamma GAE with the per-sojourn SMDP discount '
                          'exp(-beta * tau). False (default) = constant `gam` GAE.')
    alg.add_argument('--test_episodes',
                     type=int,
                     default=10,
                     help='episodes averaged per greedy (deterministic) eval point during '
                          'training (test_in_train). On deterministic envs all episodes are '
                          'identical so this is a no-op; on stochastic envs raise it for '
                          'paper-grade eval curves.')
    alg.add_argument('--eval_seed',
                     type=int,
                     default=None,
                     help='base seed pinning the greedy-eval scenarios (common random numbers): '
                          'eval episode i always runs scenario eval_seed+i, so every eval point, '
                          'every epoch and every method scores on the SAME fixed scenario set. '
                          'The surrounding RNG state is saved and restored, so training is '
                          'unaffected. Default None = historical behaviour (fresh scenarios drawn '
                          'from the live stream; measured +-0.231 SD per 20-episode point on s1, '
                          'and an inflated greedy_drift).')

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
                        default=None,
                        help='early-stopping limit on the mean per-state KL(old||new), computed '
                             'exactly over each state\'s own action set (valid for variable '
                             '|A(s)|; checked after each inner policy epoch). Default: disabled. '
                             '~0.15 stops the rare catastrophic updates that collapse a '
                             'converged policy without braking normal learning.')
    policy.add_argument('--lr_schedule',
                        type=lambda x: str(x).lower() == 'true',
                        default=False,
                        help='whether to use cosine annealing learning rate schedule')
    policy.add_argument('--normalize_advantages',
                        type=lambda x: str(x).lower() == 'true',
                        default=True,
                        help='per-batch advantage normalization (zero-mean/unit-variance). '
                             'Default True (standard PPO): needed to learn low-margin tasks. '
                             'The post-peak drift it can cause is handled by best-checkpoint '
                             'restore. Set False only for ablation.')
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
    train.add_argument('--flat_obs',
                       type=lambda x: str(x).lower() == 'true',
                       default=False,
                       help='convert observations to flat graphs (gympn/flat_graph.py) when they are '
                            'built; needs a flat encoder (type_embed / aepn) and the PPO or nfgae path')
    train.add_argument('--local_obs',
                       type=lambda x: str(x).lower() == 'true',
                       default=False,
                       help='nfgae only: component turns, so each decision observes and encodes only '
                            'its own component (GymProblem.component_turns); needs no global postpone')
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

    logging = parser.add_argument_group('logging', 'Weights & Biases logging')
    logging.add_argument('--use_wandb',
                         type=lambda x: str(x).lower() == 'true',
                         default=True,
                         help='whether to use Weights & Biases for logging')
    logging.add_argument('--wandb_mode',
                         type=str,
                         choices=['online', 'offline', 'disabled'],
                         default='offline',
                         help='W&B mode: online (cloud sync), offline (local only), or disabled')
    logging.add_argument('--wandb_project',
                         type=str,
                         default='gympn-training',
                         help='W&B project name')
    logging.add_argument('--wandb_entity',
                         type=lambda x: x if x.lower() != 'none' else None,
                         default=None,
                         help='W&B entity (username/team)')
    logging.add_argument('--open_wandb',
                         type=lambda x: str(x).lower() == 'true',
                         default=True,
                         help='whether to automatically open W&B dashboard')

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

    # Additional arguments
    train.add_argument('--num_workers',
                       type=int,
                       default=1,
                       help='number of workers for parallel episode collection')

    return parser



def make_env(args, aepn = None):
    """
    Return the training environment for this run.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    aepn : GymProblem
        The Petri net problem instance to train on.

    Returns
    ----------
    env : AEPN_Env
        The training environment.

    """
    if args.environment == 'ActionEvolutionPetriNetEnv':
        env = AEPN_Env(aepn)
    else:
        raise Exception("Unknown environment! Are you sure it is spelled correctly?")
    #env.seed(args.env_seed)
    env.reset()
    return env


def make_policy_network(args, metadata=None):
    """
    Return the policy network for this run.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    metadata : dict
        Metadata for the heterogeneous graph.

    Returns
    ----------
    policy_network : torch.nn.Module
        The policy network.

    """
    if args.environment == 'ActionEvolutionPetriNetEnv':
        if args.load_policy_network:
            policy_network = torch.load(os.path.join(os.getcwd(), args.logdir, args.name, args.policy_network))
        elif args.environment == 'ActionEvolutionPetriNetEnv':
            policy_network = HeteroActor(
                input_size=args.policy_kwargs.get("input_size", -1),
                hidden_size=args.policy_kwargs.get("hidden_size", 64),
                num_layers=args.policy_kwargs.get("num_layers", 3),
                metadata=metadata,
                num_heads=args.policy_kwargs.get("num_heads", 1),
                # Default 0.0: on-policy RL collects old log-probs in eval() (dropout
                # off) but updates in train() (dropout on). Non-zero dropout makes the
                # PPO ratio ≠ 1 even with unchanged weights → a permanent noise floor
                # that prevents the policy from settling. See INSTABILITY_ANALYSIS.md.
                dropout=args.policy_kwargs.get("dropout", 0.0),
                residual=args.policy_kwargs.get("residual", True),
                encoder=args.policy_kwargs.get("encoder", "aepn"),
                encoder_kwargs=args.policy_kwargs.get("encoder_kwargs"),
            )
        else:
            raise Exception("No policy network to load!")
    else:
        raise Exception("Unknown environment! Are you sure it is spelled correctly?")
    return policy_network


def make_value_network(args, metadata=None):
    """
    Return the value network for this run.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    metadata : dict
        Metadata for the heterogeneous graph.

    Returns
    ----------
    value_network : torch.nn.Module or None
        The value network, or None if no value network is used.
    """
    if args.value_model == 'none':
        value_network = None
    elif args.environment == 'ActionEvolutionPetriNetEnv':
        value_network = HeteroCritic(
            input_size=args.value_kwargs.get("input_size", -1),
            hidden_size=args.value_kwargs.get("hidden_size", 64),
            output_size=args.value_kwargs.get("output_size", 64),
            # num_layers/residual are the two biggest cost knobs for the HGT critic.
            # Defaults match the historical hard-coded behaviour (L3, residual on) so
            # existing configs are unchanged; pass value_kwargs to shrink the critic.
            num_layers=args.value_kwargs.get("num_layers", 3),
            num_heads=args.value_kwargs.get("num_heads", 1),
            # Default 0.0 for the same eval/train consistency reason as the policy:
            # the critic's rollout values (eval, dropout off) feed GAE, but its
            # targets are regressed in train (dropout on). See INSTABILITY_ANALYSIS.md.
            dropout=args.value_kwargs.get("dropout", 0.0),
            residual=args.value_kwargs.get("residual", True),
            encoder=args.value_kwargs.get("encoder", "aepn"),
            encoder_kwargs=args.value_kwargs.get("encoder_kwargs"),
            metadata=metadata
        )
    else:
        raise Exception("Unknown environment! Are you sure it is spelled correctly?")
    if args.value_weights != "":
        value_network.load_weights(os.path.join(os.getcwd(), args.logdir, args.name, args.value_weights))
    return value_network


def make_agent(args, metadata=None):
    """Return the agent for this run.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    metadata : dict
        Metadata for the heterogeneous graph.

    Returns
    ----------
    agent : PGAgent (experimental) or PPOAgent
        The agent.
    """
    policy_network = make_policy_network(args, metadata=metadata)
    value_network = make_value_network(args, metadata=metadata)

    common = dict(policy_lr=args.policy_lr, policy_updates=args.policy_updates,
                  value_network=value_network, value_lr=args.value_lr, value_updates=args.value_updates,
                  gam=args.gam, lam=args.lam, kld_limit=args.policy_kld_limit, ent_bonus=args.ent_bonus,
                  beta=getattr(args, 'beta', 0.0),
                  smdp_discount=getattr(args, 'smdp_discount', False),
                  nfgae=getattr(args, 'nfgae', False),
                  normalize_advantages=getattr(args, 'normalize_advantages', False),
                  lr_schedule=getattr(args, 'lr_schedule', True))
    if args.algorithm == 'pg':
        agent = PGAgent(policy_network=policy_network, **common)
    elif args.algorithm == 'ppo-clip':
        agent = PPOAgent(policy_network=policy_network, method='clip', eps=args.eps, **common)
    elif args.algorithm == 'ppo-penalty':
        agent = PPOAgent(policy_network=policy_network, method='penalty', c=args.c, **common)
    else:
        raise Exception("Unknown algorithm! Are you sure it is spelled correctly?")

    # Carried so Agent.train can pin the initial policy to the seed alone
    # (gympn.seeding.seed_network_init); without it, how many RNG draws are
    # consumed before the lazy parameters materialise varies by method and by
    # run, so two arms of one experiment do not share an initial policy.
    agent.agent_seed = getattr(args, 'agent_seed', None)

    return agent


def make_logdir(args):
    """Return the directory name for this run.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.

    Returns
    ----------
    logdir : str
        The directory name for this run.
    """
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
    tensorboard_process : subprocess.Popen or None
        The TensorBoard process running in the background, or None if launch failed.
    """
    import socket

    def is_port_available(p):
        """Check if a port is available."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', p))
            sock.close()
            return result != 0
        except:
            return True

    # Find an available port if the default one is busy
    original_port = port
    while not is_port_available(port) and port < original_port + 100:
        print(f"⚠ Port {port} is in use, trying {port + 1}...")
        port += 1

    if not is_port_available(port):
        print(f"✗ Could not find available port starting from {original_port}")
        return None

    try:
        import urllib.request
        import urllib.error

        # Start TensorBoard as a subprocess
        print(f"ℹ Starting TensorBoard on port {port}...")

        tensorboard_process = subprocess.Popen(
            [
                sys.executable, "-m", "tensorboard.main",
                "--logdir", logdir,
                "--port", str(port),
                "--reload_interval", str(reload_interval),
                "--bind_all"  # Listen on all interfaces
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        # Wait for TensorBoard to start
        print(f"ℹ Waiting {wait_time}s for TensorBoard to start...")
        time.sleep(wait_time)

        # Check if process is still running
        if tensorboard_process.poll() is not None:
            # Process exited, get error output
            _, stderr = tensorboard_process.communicate()
            print(f"✗ TensorBoard failed to start!")
            print(f"Error: {stderr}")
            return None

        # Verify TensorBoard is actually listening on the port
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = urllib.request.urlopen(f"http://localhost:{port}", timeout=2)
                if response.status == 200:
                    print(f"✓ TensorBoard is running on http://localhost:{port}")
                    break
            except (urllib.error.URLError, OSError):
                if attempt < max_retries - 1:
                    print(f"ℹ Checking TensorBoard... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(1)
                else:
                    print(f"⚠ Could not verify TensorBoard started after {max_retries} attempts")
                    print(f"  Trying anyway: http://localhost:{port}")

        # Open TensorBoard in the default web browser
        try:
            webbrowser.open(f"http://localhost:{port}", new=0, autoraise=False)
            print(f"✓ Browser opened: http://localhost:{port}")
        except Exception as e:
            print(f"⚠ Could not open browser: {e}")
            print(f"  Open manually: http://localhost:{port}")

        return tensorboard_process

    except Exception as e:
        print(f"✗ Failed to launch TensorBoard: {e}")
        import traceback
        traceback.print_exc()
        return None
