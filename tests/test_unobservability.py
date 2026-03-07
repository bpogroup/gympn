import copy
import unittest

from simpn.simulator import SimToken

from gympn.simulator import GymProblem


class TestUnobservableTokenAttrs(unittest.TestCase):
    """Tests that unobservable token attributes are properly hidden from observations."""

    def _make_problem(self):
        """Create a simple problem with two places and one action."""
        problem = GymProblem(tag='a')
        arrival = problem.add_var("arrival", var_attributes=['id', 'secret'])
        waiting = problem.add_var("waiting", var_attributes=['id', 'secret'])
        employee = problem.add_var("employee", var_attributes=['code_employee'])
        busy = problem.add_var("busy", var_attributes=['id', 'code_employee'])

        arrival.put({'id': 0, 'secret': 42})
        waiting.put({'id': 1, 'secret': 99})
        employee.put({'code_employee': 1})

        def arrive(a):
            return [SimToken({'id': a['id'] + 1, 'secret': a['secret']}, delay=1),
                    SimToken({'id': a['id'], 'secret': a['secret']})]

        problem.add_event([arrival], [arrival, waiting], arrive)

        def start(c, r):
            return [SimToken((c, r), delay=1)]

        problem.add_action([waiting, employee], [busy], behavior=start)
        return problem

    # ------------------------------------------------------------------
    # Heuristic observation tests
    # ------------------------------------------------------------------

    def test_heuristic_obs_hides_unobservable_token_attrs(self):
        """Token values in the heuristic observation must not contain the hidden attribute."""
        problem = self._make_problem()
        problem.set_unobservable(token_attrs={'arrival': ['secret'], 'waiting': ['secret']})

        obs = problem.get_heuristic_observation()

        # var_attributes should no longer list 'secret'
        self.assertNotIn('secret', obs.var_attributes.get('arrival', []))
        self.assertNotIn('secret', obs.var_attributes.get('waiting', []))

        # actual token values must also be stripped
        for p in obs.places:
            for token in p.marking:
                if isinstance(token.value, dict):
                    self.assertNotIn('secret', token.value,
                                     f"Unobservable attr 'secret' leaked in place {p._id}")

    def test_heuristic_obs_preserves_observable_attrs(self):
        """Observable attributes must still be present after hiding unobservable ones."""
        problem = self._make_problem()
        problem.set_unobservable(token_attrs={'arrival': ['secret'], 'waiting': ['secret']})

        obs = problem.get_heuristic_observation()

        # 'id' should still be in var_attributes
        self.assertIn('id', obs.var_attributes.get('arrival', []))
        self.assertIn('id', obs.var_attributes.get('waiting', []))

        # token values should still have 'id'
        for p in obs.places:
            if p._id in ('arrival', 'waiting'):
                for token in p.marking:
                    if isinstance(token.value, dict):
                        self.assertIn('id', token.value)

    def test_heuristic_obs_does_not_mutate_original(self):
        """Getting a heuristic observation must NOT modify the original problem's tokens."""
        problem = self._make_problem()
        problem.set_unobservable(token_attrs={'arrival': ['secret'], 'waiting': ['secret']})

        # Capture original token values
        original_arrival_tokens = [
            dict(t.value) for t in problem.id2node['arrival'].marking if isinstance(t.value, dict)
        ]

        _ = problem.get_heuristic_observation()

        # Original tokens must still have 'secret'
        for t in problem.id2node['arrival'].marking:
            if isinstance(t.value, dict):
                self.assertIn('secret', t.value,
                              "Original token was mutated by get_heuristic_observation!")

        current = [dict(t.value) for t in problem.id2node['arrival'].marking if isinstance(t.value, dict)]
        self.assertEqual(original_arrival_tokens, current)

    # ------------------------------------------------------------------
    # Unobservable SimVars tests
    # ------------------------------------------------------------------

    def test_heuristic_obs_excludes_unobservable_simvars(self):
        """Entire unobservable SimVars must be absent from the observation."""
        problem = self._make_problem()
        problem.set_unobservable(simvars=['employee'])

        obs = problem.get_heuristic_observation()
        obs_place_ids = [p._id for p in obs.places]

        self.assertNotIn('employee', obs_place_ids)
        # Other places should still be present
        self.assertIn('arrival', obs_place_ids)
        self.assertIn('waiting', obs_place_ids)

    # ------------------------------------------------------------------
    # set_unobservable validation tests
    # ------------------------------------------------------------------

    def test_set_unobservable_rejects_invalid_place(self):
        """Setting unobservable attrs for a non-existent place must raise."""
        problem = self._make_problem()
        with self.assertRaises(Exception):
            problem.set_unobservable(token_attrs={'nonexistent': ['id']})

    def test_set_unobservable_rejects_invalid_attr(self):
        """Setting an attribute that does not exist on the place must raise."""
        problem = self._make_problem()
        with self.assertRaises(Exception):
            problem.set_unobservable(token_attrs={'arrival': ['nonexistent_attr']})

if __name__ == '__main__':
    unittest.main()

