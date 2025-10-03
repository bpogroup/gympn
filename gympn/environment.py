import copy
from gymnasium import spaces, Env

class AEPN_Env(Env):
    """
    Gym environment for training a Deep Reinforcement Learning agent on the AEPN simulator.

    Attributes
    ----------
    pn : GymProblem
        The Petri net problem instance.
    frozen_pn : GymProblem
        A deep copy of the initial Petri net state.
    metadata : dict
        Environment metadata.
    action_space : gymnasium.spaces.Discrete
        The action space of the environment.
    observation_space : gymnasium.spaces.Dict
        The observation space of the environment.
    run : list
        List to track the simulation run.
    i : int
        Current step counter.
    active_model : bool
        Flag indicating if the model is active.
    """

    def __init__(self, aepn):
        """"
        Initialization function for the environment.
        :param aepn: the GymProblem instance to train on.
        """
        super().__init__()
        self.pn = aepn
        self.frozen_pn = copy.deepcopy(self.pn)
        self.metadata = None

        # gym specific
        # Define the action space as the nodes in the graph (currently unused)
        self.action_space = spaces.Discrete(1)

        # Define the observation space (currently unused)
        self.observation_space = spaces.Dict(
            {
                'graph': spaces.Box(low=0, high=1, shape=(1,)),
            }
        )

        # mimic the network's organization
        self.run = []
        self.i = 0
        self.active_model = True

    def step(self, action):
        """
            Execute one step in the environment.

            Parameters
            ----------
            action : int
                The action to take, must be between 0 and len(self.pn.pn_actions)-1.

            Returns
            -------
            tuple
                Contains (observation, reward, terminated, truncated, info) where:
                - observation: The current state of the environment
                - reward: The reward obtained from the action
                - terminated: Whether the episode has ended
                - truncated: Whether the episode was artificially terminated
                - info: Additional information (dictionary with 'pn_reward')
        """

        old_rewards = self.pn.reward
        if action < 0 or action >= len(self.pn.pn_actions):
            raise ValueError(f"Action {action} is not valid. Must be between 0 and {len(self.pn.pn_actions)-1}")

        binding = self.pn.pn_actions[action]
        self.pn.fire(binding) #the third value is priority (highest for sinle assignment)
        if binding[-1]._id in self.pn.reward_functions.keys():
            self.pn.update_reward(binding)
        self.pn.bindings() #updates the network tag if needed

        observation, terminated, self.i = self.pn.run_evolutions(self.run, self.i, self.active_model)

        if terminated: print(f'Terminated at time {self.pn.clock}')

        reward = (self.pn.reward - old_rewards)  # /(1+(self.pn.clock - old_clock))

        # print(f"Action taken: {action}, corresponding to binding: {binding}, generated reward: {reward}")
        info = {'pn_reward': self.pn.reward}
        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.

        Parameters
        ----------
        :param seed: Optional seed for random number generation.
        :param options: Optional dictionary of options for resetting the environment.

        Returns
        -------
        :return: The initial observation of the environment.
        """

        print(f"Entered reset with current reward for PN: {self.pn.reward} \n")
        self.pn = copy.deepcopy(self.frozen_pn)
        if self.pn.network_tag.is_evolution():
            self.pn.get_to_first_action()
        observation = self.pn.get_graph_observation()
        if self.metadata is None:
            self.metadata = self.pn.metadata
        return observation
