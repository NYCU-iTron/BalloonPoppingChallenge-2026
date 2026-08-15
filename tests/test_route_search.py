import unittest

from scripts.search_itron_routes import (
    RouteTask,
    launch_steps,
    parse_corridors,
    result_sort_key,
    task_key,
)


class TestRouteSearch(unittest.TestCase):
    def test_launch_range_is_converted_to_exact_simulation_steps(self):
        self.assertEqual(launch_steps(0.01, 10.0, 10.5, 0.25), [1000, 1025, 1050])

    def test_task_key_changes_with_route_or_implementation(self):
        base = RouteTask(1, 0, 1000, 10.0, (1, 2), "/cache", "p", "code-a")
        route_changed = RouteTask(1, 0, 1000, 10.0, (2, 1), "/cache", "p", "code-a")
        code_changed = RouteTask(1, 0, 1000, 10.0, (1, 2), "/cache", "p", "code-b")
        guidance_changed = RouteTask(
            1, 0, 1000, 10.0, (1, 2), "/cache", "p", "code-a", 0.2
        )
        controller_changed = RouteTask(
            1,
            0,
            1000,
            10.0,
            (1, 2),
            "/cache",
            "p",
            "code-a",
            controller_feedforward=0.5,
        )

        self.assertNotEqual(task_key(base), task_key(route_changed))
        self.assertNotEqual(task_key(base), task_key(code_changed))
        self.assertNotEqual(task_key(base), task_key(guidance_changed))
        self.assertNotEqual(task_key(base), task_key(controller_changed))

    def test_results_rank_score_before_last_pop_time(self):
        results = [
            {"popped_count": 5, "last_pop_time": 30.0},
            {"popped_count": 6, "last_pop_time": 60.0},
            {"popped_count": 6, "last_pop_time": 40.0},
        ]

        ranked = sorted(results, key=result_sort_key)

        self.assertEqual(ranked[0], {"popped_count": 6, "last_pop_time": 40.0})
        self.assertEqual(ranked[1], {"popped_count": 6, "last_pop_time": 60.0})

    def test_corridor_pairs_are_parsed_for_route_tasks(self):
        self.assertEqual(parse_corridors(["83:40", "97:12"]), ((83, 40), (97, 12)))

        with self.assertRaisesRegex(ValueError, "expected COMMITTED:FLYBY"):
            parse_corridors(["83"])


if __name__ == "__main__":
    unittest.main()
