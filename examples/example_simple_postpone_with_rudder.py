"""
RUDDER integration example with example_simple_postpone.
Shows how to add RUDDER credit assignment to your existing environment.
"""
import copy
import os
from simpn.simulator import SimToken
from gympn.simulator import GymProblem
from gympn.solvers import GymSolver, RandomSolver, HeuristicSolver
from gympn.rudder import RUDDERAgent
from gympn.visualisation import Visualisation
from gympn.logging_utils import get_logger

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == "__main__":

    # Initialize logger
    logger = get_logger(verbose=1)

    ###########################################################################
    # Run configurations
    train = True
    run_name = '2026-01-06-15-14-14_run'

    weights_path = os.path.join(os.getcwd(), "data", "train", run_name, f"best_policy.pth")

    ###########################################################################

    # Instantiate a simulation problem
    agency = GymProblem(allow_postpone=True, causal_rl=False)

    # Define cases
    arrival = agency.add_var("arrival", var_attributes=['task_type'])
    waiting = agency.add_var("waiting", var_attributes=['task_type'])
    busy = agency.add_var("busy", var_attributes=['task_type', 'resource_id'])
    arrival.put({'task_type': 0})
    arrival.put({'task_type': 0})

    # Define resources
    employee = agency.add_var("employee", var_attributes=['code_employee'])
    employee.put({'code_employee': 0})
    employee.put({'code_employee': 1})

    # Define events
    def arrive(a):
        return [SimToken(a, delay=1), SimToken(a)]
    agency.add_event([arrival], [arrival, waiting], arrive)

    def start(c, r):
        if r['code_employee'] == 1:
            return [SimToken((c, r), delay=20)]
        else:
            return [SimToken((c, r), delay=0.5)]

    agency.add_action([waiting, employee], [busy], behavior=start, name="start")

    def complete(b):
        return [SimToken(b[1])]

    def r_function(x):
        if x[1]['code_employee'] == 1:
            return 1
        else:
            return 1

    agency.add_event([busy], [employee], complete, name='complete', reward_function=r_function)

    ###########################################################################
    # RUDDER CONFIGURATION
    ###########################################################################

    # Create RUDDER agent for credit assignment
    rudder_agent = RUDDERAgent(
        state_dim=128,              # Must match your GNN observation size
        hidden_dim=256,             # Larger network for better predictions
        learning_rate=1e-3,
        device='cpu',               # Use 'cuda' if you have GPU
        training_frequency=1,       # Train RUDDER every epoch
        redistribution_method='contribution'  # or 'direct'
    )

    ###########################################################################
    # PPO + RUDDER CONFIGURATION
    ###########################################################################

    ppo_rudder_config = {
        # Algorithm selection
        "algorithm": "ppo-clip",
        "causal_rl": False,
        "enable_rudder": True,           # Enable RUDDER
        "rudder_agent": rudder_agent,    # Pass RUDDER agent

        # Hyperparameters
        "gam": 1,
        "lam": 0.99,
        "eps": 0.15,
        "c": 0.2,
        "vf_coeff": 0.1,
        "ent_bonus": 0.005,
        "agent_seed": None,

        # Policy Model
        "policy_model": "gnn",
        "policy_kwargs": {"hidden_layers": [128, 64]},
        "policy_lr": 8e-4,
        "policy_updates": 5,
        "policy_kld_limit": 0.1,
        "policy_weights": "",
        "policy_network": "",
        "score": False,
        "score_weight": 1e-3,

        # Value Model
        "value_model": "gnn",
        "value_kwargs": {"hidden_layers": [128, 64]},
        "value_lr": 8e-4,
        "value_updates": 10,
        "value_weights": "",

        # Training Parameters
        "episodes": 20,
        "epochs": 100,
        "max_episode_length": None,
        "batch_size": 32,
        "sort_states": False,
        "use_gpu": False,
        "load_policy_network": False,
        "verbose": 1,

        # Saving Parameters
        "name": "ppo_rudder_run",
        "datetag": True,
        "logdir": "data/train",
        "save_freq": 1,
        "open_tensorboard": True,

        # W&B Settings
        "use_wandb": False,
    }

    ###########################################################################
    # BASELINE PPO CONFIGURATION (for comparison)
    ###########################################################################

    ppo_baseline_config = {
        "algorithm": "ppo-clip",
        "causal_rl": False,
        "enable_rudder": False,  # Disabled for baseline

        "gam": 1,
        "lam": 0.99,
        "eps": 0.15,
        "c": 0.2,
        "vf_coeff": 0.1,
        "ent_bonus": 0.005,
        "agent_seed": None,

        "policy_model": "gnn",
        "policy_kwargs": {"hidden_layers": [128, 64]},
        "policy_lr": 8e-4,
        "policy_updates": 5,
        "policy_kld_limit": 0.1,
        "policy_weights": "",
        "policy_network": "",
        "score": False,
        "score_weight": 1e-3,

        "value_model": "gnn",
        "value_kwargs": {"hidden_layers": [128, 64]},
        "value_lr": 8e-4,
        "value_updates": 10,
        "value_weights": "",

        "episodes": 20,
        "epochs": 100,
        "max_episode_length": None,
        "batch_size": 32,
        "sort_states": False,
        "use_gpu": False,
        "load_policy_network": False,
        "verbose": 1,

        "name": "ppo_baseline_run",
        "datetag": True,
        "logdir": "data/train",
        "save_freq": 1,
        "open_tensorboard": False,

        "use_wandb": False,
    }

    ###########################################################################
    # Heuristic definition
    ###########################################################################

    def perfect_heuristic(observable_net, tokens_comb):
        """Perfect heuristic: assign task type 0 to resource 0 (faster)."""
        for k, el in tokens_comb.items():
            for binding in el:
                task = binding[0][1].value
                resource = binding[1][1].value
                if resource['code_employee'] == 0:
                    return {k: binding}
        return 'postpone'


    if train:
        # ======================================================================
        # OPTION 1: Train with RUDDER
        # ======================================================================
        logger.info("Training PPO + RUDDER for credit assignment")
        logger.info("This combines RUDDER's learned reward redistribution with PPO")
        logger.info("Expected: Faster convergence and better final performance")

        agency_rudder = GymProblem(allow_postpone=True, causal_rl=False)

        # Setup (same as original)
        arrival = agency_rudder.add_var("arrival", var_attributes=['task_type'])
        waiting = agency_rudder.add_var("waiting", var_attributes=['task_type'])
        busy = agency_rudder.add_var("busy", var_attributes=['task_type', 'resource_id'])
        arrival.put({'task_type': 0})
        arrival.put({'task_type': 0})

        employee = agency_rudder.add_var("employee", var_attributes=['code_employee'])
        employee.put({'code_employee': 0})
        employee.put({'code_employee': 1})

        agency_rudder.add_event([arrival], [arrival, waiting], arrive)
        agency_rudder.add_action([waiting, employee], [busy], behavior=start, name="start")
        agency_rudder.add_event([busy], [employee], complete, name='complete', reward_function=r_function)

        # Train with RUDDER
        history_rudder = agency_rudder.training_run(length=10, args_dict=ppo_rudder_config)

        logger.info("\n" + "="*70)
        logger.info("PPO + RUDDER training completed!")
        logger.info("="*70)

        # ======================================================================
        # OPTION 2: Train baseline PPO (for comparison)
        # ======================================================================
        logger.info("\nTraining PPO (baseline) for comparison")
        logger.info("This is standard PPO without any credit assignment enhancement")

        agency_baseline = GymProblem(allow_postpone=True, causal_rl=False)

        # Setup (same)
        arrival = agency_baseline.add_var("arrival", var_attributes=['task_type'])
        waiting = agency_baseline.add_var("waiting", var_attributes=['task_type'])
        busy = agency_baseline.add_var("busy", var_attributes=['task_type', 'resource_id'])
        arrival.put({'task_type': 0})
        arrival.put({'task_type': 0})

        employee = agency_baseline.add_var("employee", var_attributes=['code_employee'])
        employee.put({'code_employee': 0})
        employee.put({'code_employee': 1})

        agency_baseline.add_event([arrival], [arrival, waiting], arrive)
        agency_baseline.add_action([waiting, employee], [busy], behavior=start, name="start")
        agency_baseline.add_event([busy], [employee], complete, name='complete', reward_function=r_function)

        # Train baseline
        history_baseline = agency_baseline.training_run(length=10, args_dict=ppo_baseline_config)

        logger.info("\n" + "="*70)
        logger.info("PPO baseline training completed!")
        logger.info("="*70)

        # Compare results
        if history_rudder and history_baseline:
            import numpy as np
            rudder_final = history_rudder.get('returns', [])[-1] if history_rudder.get('returns') else 0
            baseline_final = history_baseline.get('returns', [])[-1] if history_baseline.get('returns') else 0

            improvement = ((rudder_final - baseline_final) / baseline_final * 100) if baseline_final > 0 else 0

            logger.info(f"\nComparison Results:")
            logger.info(f"  PPO (baseline): {baseline_final:.4f}")
            logger.info(f"  PPO + RUDDER:   {rudder_final:.4f}")
            logger.info(f"  Improvement:    {improvement:+.1f}%")

    else:
        logger.debug("Evaluation mode - test a trained model")
        # Add evaluation code here if needed


# =============================================================================
# KEY POINTS ABOUT RUDDER
# =============================================================================

"""
1. WHAT IS RUDDER?
   - Learns to predict episode returns from state sequences
   - Uses this to identify which steps are important
   - Redistributes rewards based on learned importance
   - No domain knowledge or token tracking required

2. HOW DOES IT HELP?
   - Better credit assignment for delayed rewards
   - Learns temporal patterns automatically
   - Works with any state representation
   - Complementary to PPO's policy gradient

3. WHEN TO USE RUDDER?
   - Delayed reward problems (rewards don't come immediately)
   - Complex task structure
   - Variable episode lengths
   - When simple PPO plateaus

4. HYPERPARAMETERS TO TUNE:
   - state_dim: Must match your observation dimension
   - hidden_dim: Increase for complex state dependencies (128→256→512)
   - learning_rate: Start at 1e-3, reduce if diverging
   - training_frequency: 1 (train every epoch) for most problems

5. EXPECTED IMPROVEMENTS:
   - On delayed reward tasks: 2-5x better convergence
   - On short-horizon tasks: 1-2x improvement or comparable
   - More stable final performance
"""

