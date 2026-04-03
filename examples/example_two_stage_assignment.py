"""
Two-stage task assignment problem with competing resources.

Each case goes through two stages (task_A → task_B) and there are two shared resources:
  - Resource 0: fast at task_A (delay=1), slow at task_B (delay=5)
  - Resource 1: slow at task_A (delay=5), fast at task_B (delay=1)

Since both stages draw from the same resource pool, the agent must learn to assign
resource 0 to task_A and resource 1 to task_B (i.e. each resource to its specialty).
A greedy policy that always picks the fastest available resource for the current stage
would starve the other stage — the agent needs to reason about both stages jointly.
"""
import copy
import os
from simpn.simulator import SimToken
from gympn.simulator import GymProblem
from gympn.solvers import GymSolver, RandomSolver, HeuristicSolver

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == "__main__":

    ###########################################################################
    # Run configurations
    train = True  # set to False to test a trained model
    run_name = 'two_stage_assignment'
    #run_name_complete = '2026-03-20-14-13-47_two_stage_assignment'
    run_name_complete = '2026-03-23-11-27-26_two_stage_assignment'
    visualize_random = False
    visualize_heuristic = False
    visualize_ppo = False

    weights_path = os.path.join(os.getcwd(), "data", "train", run_name_complete, "best_policy.pth")
    ###########################################################################

    # Instantiate a simulation problem.
    # For debugging RL behavior disable causal RL so advantage normalization remains enabled
    # (causal RL alters advantage normalization and credit assignment and can mask issues).
    agency = GymProblem(allow_postpone=True, causal_rl=True)

    # Define case variables (two stages).
    arrival = agency.add_var("arrival", var_attributes=['case_id'])
    waiting_A = agency.add_var("waiting_A", var_attributes=['case_id'])
    busy_A = agency.add_var("busy_A", var_attributes=['case_id', 'resource_id'])
    waiting_B = agency.add_var("waiting_B", var_attributes=['case_id'])
    busy_B = agency.add_var("busy_B", var_attributes=['case_id', 'resource_id'])

    # Seed two initial cases.
    arrival.put({'case_id': 0})
    arrival.put({'case_id': 0})

    # Define shared resource pool (two competing resources).
    resource = agency.add_var("resource", var_attributes=['resource_id'])
    resource.put({'resource_id': 0})
    resource.put({'resource_id': 1})

    # ── Events & Actions ────────────────────────────────────────────────────

    # Cases arrive periodically.
    def arrive(a):
        next_case = {'case_id': a['case_id']}  # schedule the next arrival (case id does not vary)
        return [SimToken(next_case, delay=1), SimToken(a)]

    agency.add_event([arrival], [arrival, waiting_A], arrive)

    # ACTION 1: assign a resource to task A.
    def start_A(c, r):
        """
        Resource 0 is fast at task A (delay=1), resource 1 is slow (delay=5).
        """
        delay = 0.5 if r['resource_id'] == 0 else 2
        return [SimToken((c, r), delay=delay)]

    agency.add_action([waiting_A, resource], [busy_A], behavior=start_A, name="start_A")

    # Task A completes → resource is returned, case moves to stage B.
    def complete_A(b):
        case, res = b
        return [SimToken(res), SimToken(case)]

    agency.add_event([busy_A], [resource, waiting_B], complete_A, name='complete_A')

    # ACTION 2: assign a resource to task B.
    def start_B(c, r):
        """
        Resource 1 is fast at task B (delay=1), resource 0 is slow (delay=5).
        """
        delay = 0.5 if r['resource_id'] == 1 else 2
        return [SimToken((c, r), delay=delay)]

    agency.add_action([waiting_B, resource], [busy_B], behavior=start_B, name="start_B")

    # Task B completes → resource is returned, reward is given.
    def complete_B(b):
        _, res = b
        return [SimToken(res)]

    agency.add_event([busy_B], [resource], complete_B, name='complete_B',
                     reward_function=lambda x: 1)

    # ── Training args ───────────────────────────────────────────────────────

    default_args = {
        # Algorithm
        "algorithm": "ppo-clip",
        "gam": 0.99,
        "lam": 0.95,
        "eps": 0.2,
        "c": 0.2,
        "ent_bonus": 0.01,
        "agent_seed": None,

        # Policy
        "policy_model": "gnn",
        "policy_kwargs": {"hidden_layers": [64]},
        "policy_lr": 3e-4,
        "policy_updates": 4,
        "policy_kld_limit": 0.2,
        "policy_weights": "",
        "policy_network": "",
        "score": False,
        "score_weight": 1e-3,

        # Value
        "value_model": "gnn",
        "value_kwargs": {"hidden_layers": [64]},
        "value_lr": 3e-4,
        "value_updates": 10,
        "value_weights": "",
        "vf_coeff": 0.5,

        # Training
        "episodes": 20,
        "epochs": 50,
        "max_episode_length": None,
        "batch_size": 64,
        "sort_states": False,
        "use_gpu": False,
        "load_policy_network": False,
        "verbose": 1,

        # Saving
        "name": run_name,
        "datetag": True,
        "logdir": "data/train",
        "save_freq": 1,
        "open_tensorboard": False,
    }

    # ── Heuristic ───────────────────────────────────────────────────────────

    def perfect_heuristic(observable_net, tokens_comb):
        """
        Optimal policy: assign resource 0 to task_A and resource 1 to task_B.
        Falls back to any available binding if the preferred resource isn't free.
        """
        fallback = 'postpone'
        for k, bindings in tokens_comb.items():
            for binding in bindings:
                res = binding[1][1].value
                # start_A → prefer resource 0, start_B → prefer resource 1
                if 'start_A' in k and res['resource_id'] == 0:
                    return {k: binding}
                if 'start_B' in k and res['resource_id'] == 1:
                    return {k: binding}
                if fallback is None:
                    fallback = {k: binding}
        return fallback

    # ── Run ──────────────────────────────────────────────────────────────────

    if train:
        agency.training_run(length=10, args_dict=default_args)

    else:
        n_runs = 10

        # Random solver
        random_rewards = []
        for _ in range(n_runs):
            frozen = copy.deepcopy(agency)
            random_rewards.append(frozen.testing_run(length=10, solver=RandomSolver()))
        avg = sum(random_rewards) / n_runs
        std = (sum(r ** 2 for r in random_rewards) / n_runs - avg ** 2) ** 0.5
        print(f"Random solver  — avg reward: {avg:.2f}, std: {std:.2f}")

        # Heuristic solver
        heuristic_rewards = []
        for _ in range(n_runs):
            frozen = copy.deepcopy(agency)
            heuristic_rewards.append(frozen.testing_run(length=10, solver=HeuristicSolver(perfect_heuristic)))
        avg = sum(heuristic_rewards) / n_runs
        std = (sum(r ** 2 for r in heuristic_rewards) / n_runs - avg ** 2) ** 0.5
        print(f"Heuristic solver — avg reward: {avg:.2f}, std: {std:.2f}")

        # DRL solver
        ppo_rewards = []
        for _ in range(n_runs):
            frozen = copy.deepcopy(agency)
            solver = GymSolver(weights_path=weights_path, metadata=agency.make_metadata())
            ppo_rewards.append(frozen.testing_run(length=10, solver=solver))
        avg = sum(ppo_rewards) / n_runs
        std = (sum(r ** 2 for r in ppo_rewards) / n_runs - avg ** 2) ** 0.5
        print(f"DRL solver     — avg reward: {avg:.2f}, std: {std:.2f}")

        # Box plot comparison
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.boxplot([random_rewards, heuristic_rewards, ppo_rewards],
                   tick_labels=['Random', 'Heuristic', 'DRL'],
                   patch_artist=True,
                   boxprops=dict(facecolor="lightblue"))
        ax.set_ylabel('Reward')
        ax.set_title('Two-Stage Assignment — Solver Comparison')
        plt.show()
