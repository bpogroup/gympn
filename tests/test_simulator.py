import unittest
from gympn.simulator import GymProblem

class TestGymProblem(unittest.TestCase):
    def test_add_var(self):
        problem = GymProblem()
        problem.add_var("place1", var_attributes=['id'])
        self.assertIn(problem.id2node["place1"], problem.places)

    def test_add_action(self):
        problem = GymProblem()
        p1 = problem.add_var("place1", var_attributes=['id'])
        p2 = problem.add_var("place2", var_attributes=['id'])
        problem.add_action(
            inflow=[p1],
            outflow=[p2],
            behavior=lambda x: [x[0] + 1],
            name="action1"
        )
        self.assertIn(problem.id2node["action1"], problem.actions)

    def test_add_event(self):
        problem = GymProblem()
        p1 = problem.add_var("place1", var_attributes=['id'])
        p2 = problem.add_var("place2", var_attributes=['id'])
        problem.add_event(
            inflow=[p1],
            outflow=[p2],
            behavior=lambda x: [x[0] - 1, x[0] + 1],
            name="event1"
        )
        self.assertIn(problem.id2node["event1"], problem.events)

if __name__ == '__main__':
    unittest.main()