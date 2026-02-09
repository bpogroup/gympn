"""
This file implements a stock trading problem where an agent learns to buy and sell stocks
to maximize profit. The agent observes stock prices and decides whether to buy, sell, or hold.

The simulation includes:
- Multiple stocks with time-varying prices
- A portfolio with cash and holdings
- Buy/sell actions with transaction costs
- A reward function based on profit realization
"""
import copy
import os
import numpy as np
from simpn.simulator import SimToken
from gympn.simulator import GymProblem
from gympn.solvers import GymSolver, RandomSolver, HeuristicSolver
from gympn.visualisation import Visualisation
import random

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def run_experiments(problem, solver, num_experiments, reporter=None, length=None):
    """Run multiple experiments and return mean and std of rewards."""
    rewards = []
    for i in range(num_experiments):
        problem_copy = copy.deepcopy(problem)
        reward = problem_copy.testing_run(solver, reporter=reporter, length=length)
        rewards.append(reward)
    return np.mean(rewards), np.std(rewards)


if __name__ == "__main__":

    ###########################################################################
    # Run configurations
    train = False
    run_name = '2026-02-04-11-43-16_stock_trading'
    test_episodes = 10
    visualize_ppo = True

    weights_path = os.path.join(os.getcwd(), "data", "train", run_name, f"best_policy.pth")

    ###########################################################################

    # Instantiate a simulation problem
    agency = GymProblem(causal_rl=train, allow_postpone=True)

    # Define variables for stock trading
    # Market state: contains all prices and portfolio info
    market_state = agency.add_var("market_state", var_attributes=[
        'stock_0_price', 'stock_1_price', 'stock_2_price',
        'stock_0_qty', 'stock_1_qty', 'stock_2_qty',
        'stock_0_cost', 'stock_1_cost', 'stock_2_cost',
        'cash'
    ])
    phase = agency.add_var("phase", var_attributes=['mode'])

    # Initialize market state
    market_state.put({
        'stock_0_price': 100.0, 'stock_1_price': 100.0, 'stock_2_price': 100.0,
        'stock_0_qty': 0, 'stock_1_qty': 0, 'stock_2_qty': 0,
        'stock_0_cost': 0.0, 'stock_1_cost': 0.0, 'stock_2_cost': 0.0,
        'cash': 10000.0
    })
    phase.put({'mode': 0})  # 0=market, 1=trade

    # Market evolution: update prices, then hand control to the agent
    def market_step(state, ph):
        """Update stock prices and switch phase to trading."""
        new_state = copy.deepcopy(state)
        for stock_id in range(3):
            key = f'stock_{stock_id}_price'
            price_change = random.choices([-1, 0, 1], weights=[0.2, 0.3, 0.5])[0]
            multiplier = random.uniform(0.98, 1.02)
            new_state[key] = max(50.0, state[key] * multiplier + price_change)
        # Keep market_state and phase aligned in time so actions can fire.
        return [SimToken(new_state, delay=1), SimToken({'mode': 1}, delay=1)]

    agency.add_event(
        [market_state, phase],
        [market_state, phase],
        market_step,
        name='market_evolution',
        guard=lambda state, ph: ph['mode'] == 0
    )


    # Buy action: agent buys stock 0 if conditions are met
    def buy_behavior(state, ph):
        """Buy one unit of stock 0 if we have enough cash."""
        new_state = copy.deepcopy(state)
        current_price = state['stock_0_price']
        transaction_cost = current_price * 0.01
        total_cost = current_price + transaction_cost

        if state['cash'] >= total_cost:
            # Buy is possible
            new_state['cash'] -= total_cost
            new_state['stock_0_qty'] += 1
            if state['stock_0_qty'] > 0:
                new_state['stock_0_cost'] = (state['stock_0_cost'] * state['stock_0_qty'] + current_price) / (state['stock_0_qty'] + 1)
            else:
                new_state['stock_0_cost'] = current_price

        return [SimToken(new_state), SimToken({'mode': 0})]

    agency.add_action(
        [market_state, phase],
        [market_state, phase],
        behavior=buy_behavior,
        name="buy",
        guard=lambda state, ph: ph['mode'] == 1,
        reward_function=lambda state, ph: 0.1 if state['cash'] >= state['stock_0_price'] * 1.01 else -0.05
    )

    # Hold action: do nothing but allow state to advance
    def hold_behavior(state, ph):
        """Hold current position."""
        return [SimToken(state), SimToken({'mode': 0})]

    agency.add_action(
        [market_state, phase],
        [market_state, phase],
        behavior=hold_behavior,
        name="hold",
        guard=lambda state, ph: ph['mode'] == 1,
        reward_function=lambda state, ph: 0.05
    )

    # Sell action: agent sells stock 0 if holdings exist
    def sell_behavior(state, ph):
        """Sell all units of stock 0 if we have any."""
        new_state = copy.deepcopy(state)

        if state['stock_0_price'] > 0 and state['stock_0_qty'] > 0:
            # Sell is possible
            transaction_cost = state['stock_0_price'] * state['stock_0_qty'] * 0.01
            proceeds = (state['stock_0_price'] * state['stock_0_qty']) - transaction_cost

            new_state['cash'] += proceeds
            new_state['stock_0_qty'] = 0
            new_state['stock_0_cost'] = 0.0

        return [SimToken(new_state), SimToken({'mode': 0})]

    def sell_reward(state, ph):
        """Reward function for sell action."""
        if state['stock_0_price'] > 0 and state['stock_0_qty'] > 0:
            transaction_cost = state['stock_0_price'] * state['stock_0_qty'] * 0.01
            proceeds = (state['stock_0_price'] * state['stock_0_qty']) - transaction_cost
            profit = proceeds - (state['stock_0_cost'] * state['stock_0_qty'])
            return max(-0.5, min(1.0, profit / 1000.0))  # Normalize profit
        return -0.05

    agency.add_action(
        [market_state, phase],
        [market_state, phase],
        behavior=sell_behavior,
        name="sell",
        guard=lambda state, ph: ph['mode'] == 1,
        reward_function=sell_reward
    )

    # Heuristic strategy
    def trading_heuristic(pn, actions_dict):
        """
        Heuristic: sell when profit is meaningful, buy only when cash is ample,
        otherwise hold. Chooses the highest-scoring available action.
        """
        def _extract_state(binding):
            # binding can be a list of (place, token) pairs or a single pair
            if isinstance(binding, (list, tuple)) and len(binding) > 0:
                token = binding[0][1] if isinstance(binding[0], (list, tuple)) else binding[1]
                return token.value if hasattr(token, "value") else token
            return {}

        best_action = None
        best_score = float("-inf")

        for action_name, bindings in actions_dict.items():
            if not bindings:
                continue
            for binding in bindings:
                state = _extract_state(binding)
                price = float(state.get('stock_0_price', 0))
                qty = float(state.get('stock_0_qty', 0))
                cost = float(state.get('stock_0_cost', 0))
                cash = float(state.get('cash', 0))

                score = 0.0
                if action_name == 'sell':
                    # Prefer selling only if profitable by at least 2%
                    if qty > 0 and price > 0 and cost > 0:
                        profit_ratio = (price - cost) / max(cost, 1e-6)
                        score = 2.0 * profit_ratio
                    else:
                        score = -1.0
                elif action_name == 'buy':
                    # Prefer buying only if we have ample cash
                    if cash >= price * 1.02:
                        score = 0.2
                    else:
                        score = -0.5
                elif action_name == 'hold':
                    # Hold is the safe default
                    score = 0.1

                if score > best_score:
                    best_score = score
                    best_action = {action_name: binding}

        return best_action

    # Default training arguments
    default_args = {
        "algorithm": "ppo-clip",
        "gam": 0.99,
        "lam": 0.95,
        "eps": 0.2,
        "c": 0.5,
        "ent_bonus": 0.01,
        "agent_seed": None,

        "policy_model": "gnn",
        "policy_kwargs": {"hidden_layers": [128, 128]},
        "policy_lr": 5e-4,
        "policy_updates": 3,
        "policy_kld_limit": 0.01,
        "policy_weights": "",
        "policy_network": "",
        "score": False,
        "score_weight": 1e-3,

        "value_model": "gnn",
        "value_kwargs": {"hidden_layers": [128, 128]},
        "value_lr": 1e-3,
        "value_updates": 10,
        "value_weights": "",

        "episodes": 20,
        "epochs": 100,
        "max_episode_length": 500,
        "batch_size": 64,
        "sort_states": False,
        "use_gpu": False,
        "load_policy_network": False,
        "verbose": 1,

        "name": "stock_trading",
        "datetag": True,
        "logdir": "data/train",
        "save_freq": 5,
        "open_tensorboard": False,

        "num_workers": 1,
    }

    # Freeze the initial problem for testing
    frozen_agency = copy.deepcopy(agency)

    if train:
        agency.training_run(length=10, args_dict=default_args)

    # Reset to frozen state
    agency = copy.deepcopy(frozen_agency)

    # Testing phase
    print("\n" + "="*70)
    print("STOCK TRADING PROBLEM - TESTING PHASE")
    print("="*70 + "\n")

    # Random solver evaluation
    print("Evaluating Random Solver...")
    random_avg, random_std = run_experiments(agency, RandomSolver(), test_episodes, length=10)
    print(f"  Random Solver:    {random_avg:.2f} ± {random_std:.2f}\n")

    # Heuristic solver evaluation
    agency = copy.deepcopy(frozen_agency)
    print("Evaluating Heuristic Solver...")
    heuristic_solver = HeuristicSolver(trading_heuristic)
    heuristic_avg, heuristic_std = run_experiments(agency, heuristic_solver, test_episodes, length=10)
    print(f"  Heuristic Solver: {heuristic_avg:.2f} ± {heuristic_std:.2f}\n")

    # DRL solver evaluation
    agency = copy.deepcopy(frozen_agency)
    print("Evaluating DRL Solver...")
    if os.path.exists(weights_path):
        drl_solver = GymSolver(weights_path=weights_path, metadata=agency.make_metadata())
        drl_avg, drl_std = run_experiments(agency, drl_solver, test_episodes, length=10)
        print(f"  DRL Solver:       {drl_avg:.2f} ± {drl_std:.2f}\n")

        if visualize_ppo:
            viz_agency = copy.deepcopy(frozen_agency)
            viz_agency.set_solver(GymSolver(weights_path=weights_path, metadata=viz_agency.make_metadata()))
            viz_agency.length = 10
            visual = Visualisation(viz_agency)
            visual.show()
    else:
        print(f"  DRL weights not found at {weights_path}\n")
        drl_avg = drl_std = 0

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Random Solver:     {random_avg:.2f} ± {random_std:.2f}")
    print(f"Heuristic Solver:  {heuristic_avg:.2f} ± {heuristic_std:.2f}")
    if os.path.exists(weights_path):
        print(f"DRL Solver:        {drl_avg:.2f} ± {drl_std:.2f}")
    print("="*70 + "\n")
