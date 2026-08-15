"""Interactive, low-clutter 3-D view of the selector's balloon field."""

from collections.abc import Callable, Iterable

import numpy as np

from BalloonPoppingGymEnv.utils.schema import Schema


class TargetSelectionVisualizer:
    """Visualize and manually mark targets while an agent is running.

    Persistent balloon labels are hidden by default; hovering still reveals a
    balloon's number and coordinates without crowding a large scenario. Set
    ``max_labels`` above zero to restore a limited set of persistent labels.
    Shift-click toggles a balloon in the manual selection, the C key clears it,
    and Space pauses or resumes the simulation. Normal Matplotlib dragging and
    scrolling rotate and zoom the 3-D scene.

    Matplotlib is imported only when the first frame is drawn, so importing an
    agent remains safe in headless evaluation jobs.
    """

    STATUS_COLORS = {0: "#8a8f98", 1: "#d946ef", 2: "#ef4444"}
    STATUS_LABELS = {0: "ground", 1: "released", 2: "popped"}

    def __init__(
        self,
        *,
        update_interval: float = 0.25,
        max_labels: int = 0,
        pick_radius: float = 12.0,
        history_length: int = 600,
        view_elevation: float = 24.0,
        view_azimuth: float = -58.0,
        on_selection_changed: Callable[[tuple[int, ...]], None] | None = None,
    ):
        if update_interval < 0:
            raise ValueError("update_interval must be non-negative")
        if max_labels < 0:
            raise ValueError("max_labels must be non-negative")
        if pick_radius <= 0:
            raise ValueError("pick_radius must be positive")
        if history_length < 1:
            raise ValueError("history_length must be at least one")

        self.update_interval = float(update_interval)
        self.max_labels = int(max_labels)
        self.pick_radius = float(pick_radius)
        self.history_length = int(history_length)
        self.view_elevation = float(view_elevation)
        self.view_azimuth = float(view_azimuth)
        self.on_selection_changed = on_selection_changed

        self._figure = None
        self._axis = None
        self._hover_annotation = None
        self._positions = np.empty((0, 3), dtype=float)
        self._status = np.empty(0, dtype=int)
        self._planned_targets = []
        self._current_target = None
        self._selected_targets = []
        self._rocket_history = []
        self._simulation_time = 0.0
        self._last_draw_time = -np.inf
        self._paused = False

    @property
    def selected_targets(self) -> tuple[int, ...]:
        """Balloon indices selected by the user, in click order."""
        return tuple(self._selected_targets)

    @property
    def is_paused(self) -> bool:
        """Whether the visualizer is currently blocking simulation progress."""
        return self._paused

    def set_paused(self, paused: bool) -> None:
        """Pause or resume simulation progress at the next visualizer update."""
        paused = bool(paused)
        if self._paused == paused:
            return
        self._paused = paused
        if self._figure is not None:
            self.draw()

    def set_selected_targets(self, indices: Iterable[int]) -> None:
        """Replace the manual selection, preserving order and removing repeats."""
        selected = []
        for raw_index in indices:
            index = int(raw_index)
            if index < 0:
                raise ValueError("target indices must be non-negative")
            if self._positions.size and index >= len(self._positions):
                raise IndexError(f"balloon index {index} is out of range")
            if index not in selected:
                selected.append(index)
        self._selected_targets = selected
        self._notify_selection_changed()
        if self._figure is not None:
            self.draw()

    def clear_selection(self) -> None:
        """Clear all manually selected balloons."""
        if not self._selected_targets:
            return
        self._selected_targets.clear()
        self._notify_selection_changed()
        if self._figure is not None:
            self.draw()

    def reset(self) -> None:
        """Clear trajectory and selection state while keeping the window open."""
        self._positions = np.empty((0, 3), dtype=float)
        self._status = np.empty(0, dtype=int)
        self._planned_targets = []
        self._current_target = None
        self._selected_targets = []
        self._rocket_history = []
        self._last_draw_time = -np.inf
        self._paused = False
        if self._figure is not None:
            self.draw()

    def update(
        self,
        observation: dict,
        *,
        planned_targets: Iterable[int] = (),
        current_target: int | None = None,
        rocket_position: np.ndarray | None = None,
        force: bool = False,
    ) -> None:
        """Record an observation and refresh the interactive view when due."""
        states = np.asarray(observation[Schema.Observation.BALLOON_STATES], dtype=float)
        status = np.asarray(
            observation[Schema.Observation.BALLOON_STATUS], dtype=int
        ).reshape(-1)
        if states.ndim != 2 or states.shape[1] < 3:
            raise ValueError("balloon_states must have shape (n, >=3)")
        if len(status) != len(states):
            raise ValueError("balloon_status and balloon_states must have equal length")

        simulation_time = float(
            observation.get(Schema.Observation.SIMULATION_TIME, self._simulation_time)
        )
        if simulation_time < self._simulation_time:
            self._rocket_history.clear()
            self._last_draw_time = -np.inf

        self._simulation_time = simulation_time
        self._positions = states[:, :3].copy()
        self._status = status.copy()
        self._planned_targets = self._valid_indices(planned_targets)
        self._current_target = self._valid_index(current_target)

        if rocket_position is None:
            sensors = np.asarray(
                observation.get(Schema.Observation.ROCKET_SENSORS, []), dtype=float
            ).reshape(-1)
            rocket_position = sensors[6:9] if sensors.size >= 9 else None
        self._record_rocket_position(rocket_position)

        if force or simulation_time - self._last_draw_time >= self.update_interval:
            self.draw()
            self._last_draw_time = simulation_time
        self._wait_while_paused()

    def draw(self) -> None:
        """Draw the latest state and process pending GUI events."""
        self._ensure_figure()
        view = (self._axis.elev, self._axis.azim, self._axis.roll)
        self._axis.clear()
        self._axis.view_init(elev=view[0], azim=view[1], roll=view[2])

        finite = np.isfinite(self._positions).all(axis=1)
        self._draw_balloons(finite)
        self._draw_rocket()
        self._draw_plan()
        self._draw_manual_selection()
        floor = self._apply_limits(finite)
        self._draw_labels_and_guides(finite, floor)
        self._decorate_axis()

        self._figure.tight_layout()
        self._figure.canvas.draw_idle()
        self._figure.canvas.flush_events()

    def show(self, *, block: bool = True) -> None:
        """Show the current figure, optionally blocking the caller."""
        self._ensure_figure()
        import matplotlib.pyplot as plt

        plt.show(block=block)

    def save(self, path, *, dpi: int = 150) -> None:
        """Save the current 3-D view for comparing selector changes."""
        self.draw()
        self._figure.savefig(path, dpi=dpi, bbox_inches="tight")

    def close(self) -> None:
        """Close the Matplotlib window owned by this visualizer."""
        if self._figure is None:
            return
        import matplotlib.pyplot as plt

        plt.close(self._figure)
        self._on_close(None)

    def _ensure_figure(self) -> None:
        if self._figure is not None:
            return

        import matplotlib.pyplot as plt

        plt.ion()
        self._figure = plt.figure(figsize=(11, 8))
        self._axis = self._figure.add_subplot(projection="3d")
        self._axis.view_init(
            elev=self.view_elevation,
            azim=self.view_azimuth,
        )
        self._figure.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._figure.canvas.mpl_connect("button_press_event", self._on_click)
        self._figure.canvas.mpl_connect("key_press_event", self._on_key)
        self._figure.canvas.mpl_connect("close_event", self._on_close)

    def _draw_balloons(self, finite: np.ndarray) -> None:
        for state in (0, 1, 2):
            indices = np.flatnonzero(finite & (self._status == state))
            if not indices.size:
                continue
            points = self._positions[indices]
            self._axis.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                s=30 if state == 1 else 22,
                c=self.STATUS_COLORS[state],
                alpha=0.9 if state == 1 else 0.4,
                edgecolors="white",
                linewidths=0.35,
                label=self.STATUS_LABELS[state],
                depthshade=True,
            )

    def _draw_rocket(self) -> None:
        if not self._rocket_history:
            return
        history = np.asarray(self._rocket_history)
        self._axis.plot(
            history[:, 0],
            history[:, 1],
            history[:, 2],
            color="#2563eb",
            linewidth=2.0,
            alpha=0.9,
            label="rocket path",
            zorder=4,
        )
        self._axis.scatter(
            *history[-1],
            marker="^",
            s=90,
            c="#1d4ed8",
            edgecolors="white",
            linewidths=0.7,
            zorder=8,
        )

    def _draw_plan(self) -> None:
        indices = [
            index
            for index in self._planned_targets
            if np.isfinite(self._positions[index]).all()
        ]
        if not indices:
            return
        points = self._positions[indices]
        if self._rocket_history:
            points = np.vstack((self._rocket_history[-1], points))
        self._axis.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            "--",
            color="#f59e0b",
            linewidth=2.0,
            label="selector plan",
            zorder=5,
        )
        targets = self._positions[indices]
        self._axis.scatter(
            targets[:, 0],
            targets[:, 1],
            targets[:, 2],
            s=85,
            facecolors="none",
            edgecolors="#f59e0b",
            linewidths=1.8,
            zorder=6,
        )

    def _draw_manual_selection(self) -> None:
        indices = [
            index
            for index in self._selected_targets
            if index < len(self._positions)
            and np.isfinite(self._positions[index]).all()
        ]
        if indices:
            points = self._positions[indices]
            self._axis.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                marker="*",
                s=180,
                c="#06b6d4",
                edgecolors="#164e63",
                linewidths=0.8,
                label="manual pick",
                zorder=8,
            )

        if self._current_target is not None:
            point = self._positions[self._current_target]
            if np.isfinite(point).all():
                self._axis.scatter(
                    *point,
                    marker="X",
                    s=120,
                    c="#fde047",
                    edgecolors="#713f12",
                    linewidths=0.8,
                    label="current target",
                    zorder=9,
                )

    def _apply_limits(self, finite: np.ndarray) -> float:
        points = self._positions[finite]
        if self._rocket_history:
            history = np.asarray(self._rocket_history)
            points = np.vstack((points, history)) if points.size else history
        if not points.size:
            return 0.0

        low = np.min(points, axis=0)
        high = np.max(points, axis=0)
        padding = np.maximum((high - low) * 0.08, 10.0)
        low -= padding
        high += padding
        self._axis.set_xlim(low[0], high[0])
        self._axis.set_ylim(low[1], high[1])
        self._axis.set_zlim(low[2], high[2])
        self._axis.set_box_aspect(np.maximum(high - low, 1.0), zoom=0.88)
        return float(low[2])

    def _draw_labels_and_guides(self, finite: np.ndarray, floor: float) -> None:
        for index in self._label_indices(finite):
            point = self._positions[index]
            self._axis.plot(
                [point[0], point[0]],
                [point[1], point[1]],
                [floor, point[2]],
                color="#64748b",
                linewidth=0.55,
                alpha=0.28,
                zorder=1,
            )
            self._axis.text(
                point[0],
                point[1],
                point[2],
                f"  {self._target_label(index)}",
                fontsize=8,
                fontweight="bold" if index in self._selected_targets else "normal",
                bbox={"boxstyle": "round,pad=0.18", "fc": "white", "alpha": 0.8},
                zorder=10,
            )

    def _decorate_axis(self) -> None:
        self._axis.set_xlabel("East (m)", labelpad=8)
        self._axis.set_ylabel("North (m)", labelpad=8)
        self._axis.set_zlabel("Altitude ASL (m)", labelpad=8)
        self._axis.set_title(
            f"Selector target view — t={self._simulation_time:.2f} s"
            f"{'  [PAUSED]' if self._paused else ''}\n"
            "Space: pause/resume · drag: rotate · scroll: zoom · "
            "Shift+click: toggle · C: clear",
            pad=16,
            color="#b91c1c" if self._paused else "black",
        )
        self._axis.grid(alpha=0.2)
        handles, labels = self._axis.get_legend_handles_labels()
        if handles:
            self._axis.legend(handles, labels, loc="upper left", fontsize=8)

        self._hover_annotation = self._axis.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox={"boxstyle": "round", "fc": "white", "alpha": 0.95},
            arrowprops={"arrowstyle": "->", "color": "#334155"},
            fontsize=8,
            zorder=20,
        )
        self._hover_annotation.set_visible(False)

    def _label_indices(self, finite: np.ndarray) -> list[int]:
        if self.max_labels == 0:
            return []

        priority = [
            *self._selected_targets,
            self._current_target,
            *self._planned_targets,
        ]
        labels = []
        for index in priority:
            if index is not None and index < len(finite) and finite[index]:
                if index not in labels:
                    labels.append(index)
            if len(labels) == self.max_labels:
                return labels

        candidates = np.flatnonzero(finite & (self._status == 1))
        for index in self._spread_indices(candidates, self.max_labels - len(labels)):
            if index not in labels:
                labels.append(index)
        return labels[: self.max_labels]

    def _spread_indices(self, candidates: np.ndarray, count: int) -> list[int]:
        """Choose spatial representatives rather than a crowded index prefix."""
        if count <= 0 or candidates.size == 0:
            return []
        if candidates.size <= count:
            return candidates.tolist()

        points = self._positions[candidates]
        span = np.ptp(points, axis=0)
        normalized = (points - np.min(points, axis=0)) / np.where(span > 0, span, 1.0)
        chosen = [int(np.argmin(normalized[:, 0] + normalized[:, 1]))]
        distances = np.full(len(points), np.inf)
        while len(chosen) < count:
            latest = normalized[chosen[-1]]
            distances = np.minimum(
                distances, np.linalg.norm(normalized - latest, axis=1)
            )
            distances[chosen] = -1.0
            chosen.append(int(np.argmax(distances)))
        return candidates[chosen].tolist()

    def _target_label(self, index: int) -> str:
        if index in self._selected_targets:
            return f"S{self._selected_targets.index(index) + 1}:#{index}"
        if index == self._current_target:
            return f"TARGET:#{index}"
        if index in self._planned_targets:
            return f"P{self._planned_targets.index(index) + 1}:#{index}"
        return f"#{index}"

    def _project_to_display(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from mpl_toolkits.mplot3d import proj3d

        points = self._positions[indices]
        x_2d, y_2d, depth = proj3d.proj_transform(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            self._axis.get_proj(),
        )
        display = self._axis.transData.transform(np.column_stack((x_2d, y_2d)))
        return display, np.asarray(depth)

    def _nearest_index(self, event) -> int | None:
        if event.inaxes is not self._axis or event.x is None or event.y is None:
            return None
        indices = np.flatnonzero(np.isfinite(self._positions).all(axis=1))
        if not indices.size:
            return None
        display, depth = self._project_to_display(indices)
        distances = np.linalg.norm(display - np.array([event.x, event.y]), axis=1)
        within_radius = np.flatnonzero(distances <= self.pick_radius)
        if not within_radius.size:
            return None
        nearest_depth = within_radius[np.argmax(depth[within_radius])]
        return int(indices[nearest_depth])

    def _on_motion(self, event) -> None:
        if self._hover_annotation is None:
            return
        self._hover_annotation.set_visible(False)
        index = self._nearest_index(event)
        if index is None:
            if self._figure is not None:
                self._figure.canvas.draw_idle()
            return

        from mpl_toolkits.mplot3d import proj3d

        position = self._positions[index]
        x_2d, y_2d, _ = proj3d.proj_transform(*position, self._axis.get_proj())
        self._hover_annotation.xy = (x_2d, y_2d)
        self._hover_annotation.set_text(
            f"#{index}  {self.STATUS_LABELS.get(int(self._status[index]), 'unknown')}\n"
            f"E {position[0]:.1f}  N {position[1]:.1f}  Z {position[2]:.1f} m"
        )
        self._hover_annotation.set_visible(True)
        self._figure.canvas.draw_idle()

    def _on_click(self, event) -> None:
        if event.button != 1 or event.key != "shift":
            return
        index = self._nearest_index(event)
        if index is None:
            return
        if index in self._selected_targets:
            self._selected_targets.remove(index)
        else:
            self._selected_targets.append(index)
        self._notify_selection_changed()
        self.draw()

    def _on_key(self, event) -> None:
        if event.key in (" ", "space"):
            self.set_paused(not self._paused)
        elif event.key and event.key.lower() == "c":
            self.clear_selection()

    def _on_close(self, _event) -> None:
        self._paused = False
        self._figure = None
        self._axis = None
        self._hover_annotation = None

    def _wait_while_paused(self) -> None:
        """Keep the GUI responsive while preventing the agent loop advancing."""
        if not self._paused or self._figure is None:
            return
        import matplotlib.pyplot as plt

        while self._paused and self._figure is not None:
            plt.pause(0.05)

    def _valid_indices(self, indices: Iterable[int]) -> list[int]:
        valid = []
        for raw_index in indices:
            index = self._valid_index(raw_index)
            if index is not None and index not in valid:
                valid.append(index)
        return valid

    def _valid_index(self, index: int | None) -> int | None:
        if index is None:
            return None
        index = int(index)
        return index if 0 <= index < len(self._positions) else None

    def _record_rocket_position(self, position) -> None:
        if position is None:
            return
        position = np.asarray(position, dtype=float).reshape(-1)
        if position.size < 3 or not np.isfinite(position[:3]).all():
            return
        if not self._rocket_history or not np.array_equal(
            self._rocket_history[-1], position[:3]
        ):
            self._rocket_history.append(position[:3].copy())
            self._rocket_history = self._rocket_history[-self.history_length :]

    def _notify_selection_changed(self) -> None:
        if self.on_selection_changed is not None:
            self.on_selection_changed(self.selected_targets)
