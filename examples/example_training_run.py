"""
Example training routing for the gym simulator.
"""
from simpn.simulator import SimToken

from gympn.simulator import GymProblem
from gympn.utils import sim_tokens_values_from_bindings, binding_from_tokens_values

def test_training():
    """
    Test a training in the Gym Simulator.
    """

    # Instantiate a simulation problem.
    agency = GymProblem(plot_observations=True)

    # Define queues and other 'places' in the process.
    arrival = agency.add_var("arrival", var_attributes=['id'])
    in_transit = agency.add_var("in_transit", var_attributes=['id'])

    waiting = agency.add_var("waiting", var_attributes=['id'])

    busy = agency.add_var("busy", var_attributes=['id', 'code_employee'])

    arrival.put({'id': 0})

    # Define resources.
    employee = agency.add_var("employee", var_attributes=['code_employee'])
    employee.put({'code_employee': 1})
    employee.put({'code_employee': 2})

    #Define which variables are observable (by default, every SimVar is observable, only unobservable if specified)
    agency.set_unobservable(simvars = ['in_transit'], token_attrs = {'arrival': ['id']})



    # Define events.

    def arrive(a):
        return [SimToken({'id': a['id'] + 1}, delay=1), SimToken({'id': a['id']})]


    agency.add_event([arrival], [arrival, in_transit], arrive)

    agency.add_event([in_transit], [waiting], lambda x: [SimToken(x, delay=1)], name="dispatch")

    def start(c, r):

        return [SimToken((c, r), delay= 1.0)]#delay=uniform(1, 2))]

    def assign_e1_if_available(token_combinations, problem):
        # here problem is not used, but it may be useful in more complex scenarios

        # sim_tokens_values_from_bindings returns a list of token values (i.e., strings) from a list of bindings
        assignable = sim_tokens_values_from_bindings(token_combinations)

        assignable = [el for el in assignable if el[0] == "e1" or el[1] == "e1"]
        if len(assignable):
            return binding_from_tokens_values(assignable[0], token_combinations)
        else:
            print("Resource e1 not available, returning a random assignment")
            return token_combinations[0]

    agency.add_action([waiting, employee], [busy], behavior=start)#, solver=HeuristicSolver(assign_e1_if_available))

    #one action sends an employee out (which is desired for employee 1)
    unavailable = agency.add_var("unavailable", var_attributes=['code_employee'])

    def get_out(x):
        return [SimToken(x, delay=1)]

    agency.add_action([employee], [unavailable], behavior=get_out, name="get_out", solver=None)

    agency.add_event([unavailable], [employee], behavior=lambda x: [SimToken(x)], name="return")

    def complete(b):
        return [SimToken(b[1])]

    def get_reward(b):
        if b[1]['code_employee'] == 1:
            return -1
        else:
            return 10

    agency.add_event([busy], [employee], complete, name='complete', reward_function=get_reward)

    # Default training arguments (change them as needed)
    default_args = {
        # Algorithm Parameters
        "algorithm": "ppo-clip",
        "gam": 0.99,
        "lam": 0.99,
        "eps": 0.2,
        "c": 0.2,
        "ent_bonus": 0.0,
        "agent_seed": None,

        # Policy Model
        "policy_model": "gnn",
        "policy_kwargs": {"hidden_layers": [64]},
        "policy_lr": 3e-4,
        "policy_updates": 2,
        "policy_kld_limit": 0.01,
        "policy_weights": "",
        "policy_network": "",
        "score": False,
        "score_weight": 1e-3,

        # Value Model
        "value_model": "gnn",
        "value_kwargs": {"hidden_layers": [64]},
        "value_lr": 3e-4,
        "value_updates": 10,
        "value_weights": "",

        # Training Parameters
        "episodes": 10,
        "epochs": 100,
        "max_episode_length": None,
        "batch_size": 64,
        "sort_states": False,
        "use_gpu": False,
        "load_policy_network": False,
        "verbose": 0,

        # Saving Parameters
        "name": "run",
        "datetag": True,
        "logdir": "data/train",
        "save_freq": 1,
        "open_tensorboard": True, # Open TensorBoard during training (defaults to False)
    }

    agency.training_run(length=11, args_dict=default_args)

if __name__ == "__main__":
    test_training()
