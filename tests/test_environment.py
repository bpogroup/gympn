import unittest

from simpn.simulator import SimToken

from gympn.environment import AEPN_Env
from gympn.simulator import GymProblem

class TestAEPNEnv(unittest.TestCase):
    def test_environment_initialization(self):
        problem = GymProblem()
        env = AEPN_Env(problem)
        self.assertIsNotNone(env)

    def test_reset(self):
        problem = GymProblem()
        waiting = problem.add_var("waiting", var_attributes=['id'])
        busy = problem.add_var("busy", var_attributes=['id', 'code_employee'])
        employee = problem.add_var("employee", var_attributes=['code_employee'])
        waiting.put({'id': 0})
        employee.put({'code_employee': 1})
        def start(c, r):
            return [SimToken((c, r), delay=1)]
        problem.add_action([waiting, employee], [busy], behavior=start)

        env = AEPN_Env(problem)
        state = env.reset()
        self.assertIsNotNone(state)

    def test_step(self):
        problem = GymProblem()
        waiting = problem.add_var("waiting", var_attributes=['id'])
        busy = problem.add_var("busy", var_attributes=['id', 'code_employee'])
        employee = problem.add_var("employee", var_attributes=['code_employee'])
        waiting.put({'id': 0})
        employee.put({'code_employee': 1})
        def start(c, r):
            return [SimToken((c, r), delay=1)]
        problem.add_action([waiting, employee], [busy], behavior=start)

        env = AEPN_Env(problem)
        state = env.reset()
        action = 0
        next_state, reward, terminated, _, info = env.step(action)
        self.assertIsNotNone(next_state)
        self.assertEqual(reward, 0)

if __name__ == '__main__':
    unittest.main()