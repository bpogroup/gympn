"""
Example demonstrating the lineage-critic fix for causal RL.

This example shows:
1. A simple task assignment problem
2. Training with causal RL enabled
3. Detailed debugging output tracking:
   - Redistributed credits per action
   - Q-values learned by the critic
   - Advantages computed as Q - V
   - Value function convergence
4. How the policy should eventually prefer high-credit actions
"""

import copy
import os
from simpn.simulator import SimToken
from gympn.simulator import GymProblem
from gympn.solvers import HeuristicSolver
from gympn.logging_utils import get_logger
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == "__main__":

    logger = get_logger(verbose=1)

    ###########################################################################
    # Configuration
    ###########################################################################
    train = True
    run_name = '2026-01-06-lineage-critic-test'

    ###########################################################################
    # Problem setup: Simple task assignment with two employees
    ###########################################################################
    agency = GymProblem(allow_postpone=True, causal_rl=True)

    # Define cases (tasks)
    arrival = agency.add_var("arrival", var_attributes=['task_type'])
    waiting = agency.add_var("waiting", var_attributes=['task_type'])
    busy = agency.add_var("busy", var_attributes=['task_type', 'resource_id'])
    arrival.put({'task_type': 0})
    arrival.put({'task_type': 0})

    # Define resources (employees)
    employee = agency.add_var("employee", var_attributes=['code_employee'])
    employee.put({'code_employee': 0})  # Fast employee
    employee.put({'code_employee': 1})  # Slow employee

    # Event: Task arrives
    def arrive(a):
        return [SimToken(a, delay=1), SimToken(a)]
    agency.add_event([arrival], [arrival, waiting], arrive)

    # Action: Assign task to employee
    def start(c, r):
        """
        Assign task c to resource r.
        Employee 0 (code_employee=0) is fast: delay=0.5
        Employee 1 (code_employee=1) is slow: delay=20

        The agent should learn to prefer assigning to employee 0.
        """
        if r['code_employee'] == 1:
            return [SimToken((c, r), delay=20)]
        else:
            return [SimToken((c, r), delay=0.5)]

    agency.add_action([waiting, employee], [busy], behavior=start, name="start")

    # Event: Task completes
    def complete(b):
        """Task completion frees up the employee."""
        return [SimToken(b[1])]

    def r_function(x):
        """Reward: 1 for each completed task."""
        return 1

    agency.add_event([busy], [employee], complete, name='complete', reward_function=r_function)

    ###########################################################################
    # Hyperparameters optimized for lineage-critic learning
    ###########################################################################

    config = {
        # Core algorithm
        "algorithm": "ppo-clip",  # Use clipped PPO
        "causal_rl": True,  # **CRITICAL**: Enable causal RL for lineage-critic fix
        "gam": 1.0,  # Full credit assignment (don't discount)
        "lam": 0.99,  # GAE discount
        "eps": 0.15,
        "c": 0.2,
        "vf_coeff": 0.5,
        "ent_bonus": 0.01,
        "agent_seed": None,

        # Policy Model
        "policy_model": "gnn",
        "policy_kwargs": {"hidden_layers": [128, 64]},
        "policy_lr": 1e-3,  # Slightly higher LR for faster convergence
        "policy_updates": 5,
        "policy_kld_limit": 0.1,

        # Value Model (Q-head)
        "value_model": "gnn",
        "value_kwargs": {"hidden_layers": [128, 64]},
        "value_lr": 5e-3,  # INCREASED from 1e-3 for faster Q-head learning
        "value_updates": 10,  # More value updates to fit Q-head

        # Training
        "episodes": 32,
        "epochs": 50,  # Longer training to see convergence
        "max_episode_length": None,
        "batch_size": 32,
        "sort_states": False,
        "use_gpu": False,
        "load_policy_network": False,
        "verbose": 1,

        # Saving
        "name": "lineage-critic-test",
        "datetag": True,
        "logdir": "data/train",
        "save_freq": 1,
        "open_tensorboard": False,

        # W&B
        "open_wandb": False,
        "wandb_mode": "offline",
        "use_wandb": False,

        # **CRITICAL**: Debug mode to track advantages, Q-values, and credits
        "debug": False,
    }

    if train:
        logger.info("="*70)
        logger.info("LINEAGE-CRITIC FIX DEMONSTRATION")
        logger.info("="*70)
        logger.info("")
        logger.info("Problem: Two employees with different processing speeds")
        logger.info("  - Employee 0 (fast): processes tasks in 0.5 time units")
        logger.info("  - Employee 1 (slow): processes tasks in 20 time units")
        logger.info("")
        logger.info("Objective: Learn to assign tasks to the faster employee")
        logger.info("")
        logger.info("Expected behavior:")
        logger.info("  - Early training: Policy explores both employees (entropy high)")
        logger.info("  - Mid training: Policy starts preferring employee 0 (lower advantage)")
        logger.info("  - Late training: Policy converges to always using employee 0")
        logger.info("")
        logger.info("With lineage-critic fix:")
        logger.info("  - Q(c, a) trained on redistributed credits")
        logger.info("  - V(c) = Sum of pi(a|c) Q(c, a)")
        logger.info("  - A(c, a) = Q(c, a) - V(c)")
        logger.info("  -> Actions with zero credit get ~zero advantage (no spurious baseline offset)")
        logger.info("="*70)
        logger.info("")

        try:
            agency.training_run(length=10, args_dict=config)
            logger.info("")
            logger.info("="*70)
            logger.info("Training completed successfully!")
            logger.info("Check the debug output above to see:")
            logger.info("  - [CAUSAL ADVANTAGES]: Advantage tracking per action")
            logger.info("  - [Q-HEAD TRAINING]: Q-value convergence")
            logger.info("  - [LINEAGE ADVANTAGE]: Q - V computation verification")
            logger.info("="*70)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.info("Training disabled. Set train=True to start training.")

