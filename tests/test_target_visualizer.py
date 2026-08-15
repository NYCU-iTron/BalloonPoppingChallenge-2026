import unittest
from unittest import mock

import numpy as np

from BalloonPoppingGymEnv.agents.gnc.target_visualizer import (
    TargetSelectionVisualizer,
)


def _observation(time=1.0):
    states = np.zeros((20, 6), dtype=float)
    states[:, 0] = np.arange(20) * 10.0
    states[:, 1] = (np.arange(20) % 5) * 15.0
    states[:, 2] = 100.0 + np.arange(20) * 3.0
    return {
        "simulation_time": time,
        "balloon_states": states,
        "balloon_status": np.ones(20, dtype=int),
        "rocket_sensors": np.array([0.0] * 6 + [5.0, 6.0, 100.0] + [0.0] * 3),
    }


class TestTargetSelectionVisualizer(unittest.TestCase):
    def test_balloon_labels_are_hidden_by_default(self):
        visualizer = TargetSelectionVisualizer()
        visualizer._positions = _observation()["balloon_states"][:, :3]
        visualizer._status = np.ones(20, dtype=int)
        visualizer._selected_targets = [17]
        visualizer._current_target = 8
        visualizer._planned_targets = [8, 4, 12]

        labels = visualizer._label_indices(np.ones(20, dtype=bool))

        self.assertEqual(labels, [])

    def test_selection_keeps_click_order_and_removes_duplicates(self):
        visualizer = TargetSelectionVisualizer()
        visualizer._positions = np.zeros((5, 3))

        visualizer.set_selected_targets([3, 1, 3])

        self.assertEqual(visualizer.selected_targets, (3, 1))

    def test_priority_targets_receive_the_limited_labels(self):
        visualizer = TargetSelectionVisualizer(max_labels=4)
        visualizer._positions = _observation()["balloon_states"][:, :3]
        visualizer._status = np.ones(20, dtype=int)
        visualizer._selected_targets = [17]
        visualizer._current_target = 8
        visualizer._planned_targets = [8, 4, 12]

        labels = visualizer._label_indices(np.ones(20, dtype=bool))

        self.assertEqual(labels, [17, 8, 4, 12])

    def test_update_throttles_drawing_but_keeps_the_latest_state(self):
        visualizer = TargetSelectionVisualizer(update_interval=1.0)
        with mock.patch.object(visualizer, "draw") as draw:
            visualizer.update(_observation(1.0), planned_targets=[2, 4])
            second = _observation(1.2)
            second["balloon_states"][0, 0] = 999.0
            visualizer.update(second, planned_targets=[5])

        draw.assert_called_once_with()
        self.assertEqual(visualizer._positions[0, 0], 999.0)
        self.assertEqual(visualizer._planned_targets, [5])

    def test_update_rejects_mismatched_status_length(self):
        observation = _observation()
        observation["balloon_status"] = np.ones(2, dtype=int)

        with self.assertRaisesRegex(ValueError, "equal length"):
            TargetSelectionVisualizer().update(observation)

    def test_selection_callback_receives_an_immutable_snapshot(self):
        calls = []
        visualizer = TargetSelectionVisualizer(on_selection_changed=calls.append)
        visualizer._positions = np.zeros((5, 3))

        visualizer.set_selected_targets([2, 4])

        self.assertEqual(calls, [(2, 4)])
        self.assertIsInstance(calls[0], tuple)

    def test_draw_creates_an_interactive_3d_scene(self):
        visualizer = TargetSelectionVisualizer()
        self.addCleanup(visualizer.close)

        visualizer.update(
            _observation(), planned_targets=[2, 4], current_target=2, force=True
        )

        self.assertEqual(visualizer._axis.name, "3d")
        self.assertEqual(visualizer._axis.get_xlabel(), "East (m)")
        self.assertEqual(visualizer._axis.get_zlabel(), "Altitude ASL (m)")

    def test_pause_state_can_be_toggled_programmatically(self):
        visualizer = TargetSelectionVisualizer()

        visualizer.set_paused(True)
        self.assertTrue(visualizer.is_paused)

        visualizer.set_paused(False)
        self.assertFalse(visualizer.is_paused)

    def test_reset_and_close_release_a_pause(self):
        visualizer = TargetSelectionVisualizer()
        visualizer._paused = True

        visualizer.reset()
        self.assertFalse(visualizer.is_paused)

        visualizer._paused = True
        visualizer._on_close(None)
        self.assertFalse(visualizer.is_paused)


if __name__ == "__main__":
    unittest.main()
