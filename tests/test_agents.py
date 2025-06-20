import unittest
from unittest.mock import MagicMock

from simpn.simulator import SimToken

from gympn.agents import Agent
from gympn.environment import AEPN_Env
from gympn.simulator import GymProblem
from gympn.networks import HeteroActor, HeteroCritic


class TestRLAgent(unittest.TestCase):
    def setUp(self):
        # Create a mock problem and environment
        self.problem = GymProblem(tag='a')
        arrival = self.problem.add_var("arrival", var_attributes=['id'])
        waiting = self.problem.add_var("waiting", var_attributes=['id'])
        busy = self.problem.add_var("busy", var_attributes=['id', 'code_employee'])
        employee = self.problem.add_var("employee", var_attributes=['code_employee'])
        arrival.put({'id': 0})
        waiting.put({'id': 0})
        employee.put({'code_employee': 1})

        # Define a simple event
        def arrive(a):
            return [SimToken({'id': a['id'] + 1}, delay=1), SimToken({'id': a['id']})]
        self.problem.add_event([arrival], [arrival, waiting], arrive)
        # Define a simple action
        def start(c, r):
            return [SimToken((c, r), delay=1)]
        self.problem.add_action([waiting, employee], [busy], behavior=start)

        self.env = AEPN_Env(self.problem)

        # Initialize the agent with the mock environment
        self.agent = Agent(policy_network=HeteroActor(metadata=self.problem.make_metadata(), output_size=1), value_network=HeteroCritic(metadata=self.problem.make_metadata(), output_size=1))

    def test_agent_initialization(self):
        self.assertEqual(self.agent.policy_model.decoder[-1].out_features, 1)
        self.assertEqual(self.agent.value_model.lin.out_features, 1)

    def test_agent_action(self):
        # Mock the agent's policy to return a fixed action
        obs = self.problem.get_graph_observation()

        # Test the agent's action in the environment
        action = self.agent.act(obs)
        self.assertEqual(action, 0)

if __name__ == '__main__':
    unittest.main()