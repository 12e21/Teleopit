#!/usr/bin/env python3
"""Web-based dataset clip review tool using viser.

Plays reference motion clips one-by-one in a browser and provides
GUI controls for annotation (keep/drop/skip, difficulty, notes).

Usage:
    python scripts/data/review_dataset.py \
        --dataset twist2_full \
        --review data/datasets/review/twist2_full/review_state.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from threading import Lock


import mujoco
import numpy as np
import viser

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from mjlab.viewer.viser import ViserMujocoScene

from train_mimic.data.review_lib import (
    ReviewRow,
    ReviewStats,
    compute_review_stats,
    load_review_state,
    save_review_state,
    utc_now_iso,
)
from train_mimic.data.review_metrics import (
    ClipMetrics,
    compute_all_metrics,
    suspicion_score,
)

DEFAULT_XML = PROJECT_ROOT / "train_mimic" / "assets" / "g1" / "g1_sim2sim_29dof.xml"


# ---------------------------------------------------------------------------
# ClipPlayer: loads NPZ clip and drives MuJoCo qpos per frame
# ---------------------------------------------------------------------------

class ClipPlayer:
    """Loads a single clip NPZ and sets MuJoCo qpos frame-by-frame."""

    def __init__(self, mj_model: mujoco.MjModel) -> None:
        self.model = mj_model
        self.data = mujoco.MjData(mj_model)
        self._joint_pos: np.ndarray | None = None  # (T, 29)
        self._pelvis_pos: np.ndarray | None = None  # (T, 3)
        self._pelvis_quat: np.ndarray | None = None  # (T, 4) wxyz
        self._fps: int = 30
        self._num_frames: int = 0

    def load_clip(self, npz_path: Path) -> None:
        """Load NPZ clip data."""
        d = np.load(str(npz_path), allow_pickle=True)
        self._joint_pos = np.asarray(d["joint_pos"])  # (T, 29)
        body_pos_w = np.asarray(d["body_pos_w"])  # (T, nb, 3)
        body_quat_w = np.asarray(d["body_quat_w"])  # (T, nb, 4)
        self._pelvis_pos = body_pos_w[:, 0, :]  # pelvis = body 0
        self._pelvis_quat = body_quat_w[:, 0, :]
        self._fps = int(d["fps"])
        self._num_frames = self._joint_pos.shape[0]

    def set_frame(self, frame_idx: int) -> None:
        """Set qpos from frame data and run mj_forward."""
        if self._joint_pos is None:
            return
        idx = max(0, min(frame_idx, self._num_frames - 1))
        self.data.qpos[:3] = self._pelvis_pos[idx]
        self.data.qpos[3:7] = self._pelvis_quat[idx]
        self.data.qpos[7:] = self._joint_pos[idx]
        mujoco.mj_forward(self.model, self.data)

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def duration_s(self) -> float:
        return self._num_frames / self._fps if self._fps > 0 else 0.0


# ---------------------------------------------------------------------------
# ReviewSession: manages review state, navigation, persistence
# ---------------------------------------------------------------------------

class ReviewSession:
    """Manages review state, navigation order, and persistence."""

    def __init__(
        self,
        review_path: Path,
        sort_mode: str = "unreviewed_first",
        metrics: dict[str, ClipMetrics] | None = None,
    ) -> None:
        self._review_path = review_path
        self._rows: list[ReviewRow] = load_review_state(review_path)
        self._metrics = metrics or {}
        self._order: list[int] = []  # indices into _rows
        self._cursor: int = 0
        self._lock = Lock()
        self._reorder(sort_mode)

    @property
    def total(self) -> int:
        return len(self._rows)

    @property
    def cursor_display(self) -> int:
        """1-based position in the current ordering."""
        return self._cursor + 1

    def current_row(self) -> ReviewRow | None:
        if not self._order:
            return None
        return self._rows[self._order[self._cursor]]

    def go_next(self) -> ReviewRow | None:
        if self._cursor < len(self._order) - 1:
            self._cursor += 1
        return self.current_row()

    def go_prev(self) -> ReviewRow | None:
        if self._cursor > 0:
            self._cursor -= 1
        return self.current_row()

    def go_next_unreviewed(self) -> ReviewRow | None:
        """Jump to the next unreviewed clip after the current cursor."""
        for i in range(self._cursor + 1, len(self._order)):
            if self._rows[self._order[i]].decision == "":
                self._cursor = i
                return self.current_row()
        # Wrap around from beginning
        for i in range(0, self._cursor):
            if self._rows[self._order[i]].decision == "":
                self._cursor = i
                return self.current_row()
        return self.current_row()

    def jump_to(self, position: int) -> ReviewRow | None:
        """Jump to a 1-based position in the ordering."""
        idx = max(0, min(position - 1, len(self._order) - 1))
        self._cursor = idx
        return self.current_row()

    def annotate(
        self,
        decision: str,
        difficulty: str = "",
        issue_tags: str = "",
        note: str = "",
    ) -> None:
        """Set annotation on current row and save to disk."""
        with self._lock:
            row = self.current_row()
            if row is None:
                return
            row.decision = decision
            row.difficulty = difficulty
            row.issue_tags = issue_tags
            row.note = note
            row.reviewed_at = utc_now_iso()
            self.save()

    def save(self) -> None:
        save_review_state(self._rows, self._review_path)

    def stats(self) -> ReviewStats:
        return compute_review_stats(self._rows)

    def get_metrics(self, clip_id: str) -> ClipMetrics | None:
        return self._metrics.get(clip_id)

    def _reorder(self, sort_mode: str) -> None:
        indices = list(range(len(self._rows)))

        if sort_mode == "unreviewed_first":
            indices.sort(key=lambda i: (
                0 if self._rows[i].decision == "" else 1,
                self._rows[i].source,
                self._rows[i].clip_id,
            ))
        elif sort_mode == "source":
            indices.sort(key=lambda i: (self._rows[i].source, self._rows[i].clip_id))
        elif sort_mode == "duration_desc":
            indices.sort(key=lambda i: -self._rows[i].duration_s)
        elif sort_mode == "suspicion_desc":
            indices.sort(key=lambda i: (
                -suspicion_score(self._metrics[self._rows[i].clip_id])
                if self._rows[i].clip_id in self._metrics else 0.0,
                self._rows[i].clip_id,
            ))
        else:
            indices.sort(key=lambda i: self._rows[i].clip_id)

        self._order = indices
        self._cursor = 0


# ---------------------------------------------------------------------------
# ReviewViewerApp: viser server + GUI + main loop
# ---------------------------------------------------------------------------

class ReviewViewerApp:
    """Main application: ties viser, ClipPlayer, and ReviewSession together."""

    def __init__(
        self,
        review_path: Path,
        xml_path: Path,
        project_root: Path,
        *,
        port: int = 8012,
        sort_mode: str = "unreviewed_first",
        metrics: dict[str, ClipMetrics] | None = None,
    ) -> None:
        self._project_root = project_root
        self._model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._player = ClipPlayer(self._model)
        self._session = ReviewSession(review_path, sort_mode, metrics)

        self._server = viser.ViserServer(port=port, label="Clip Review")
        self._scene = ViserMujocoScene.create(
            server=self._server, mj_model=self._model, num_envs=1,
        )

        # Pre-warm viser's cached_property type hint resolution at a shallow
        # call stack.  Without this, the first update_from_mjdata triggered
        # from a deep callback chain causes RecursionError in Python 3.10's
        # get_type_hints / _eval_type.
        mujoco.mj_forward(self._model, self._player.data)
        self._scene.update_from_mjdata(self._player.data)

        # Playback state
        self._playing: bool = False
        self._speed: float = 1.0
        self._current_frame: int = 0

        # Pending action queue: callbacks only append here, main loop processes.
        # This avoids deep recursion inside viser's callback / message queue.
        self._pending_actions: list[str] = []
        self._pending_frame_scrub: int | None = None  # from slider drag
        self._pending_jump: int | None = None  # from jump input

    def setup_gui(self) -> None:
        """Build all viser GUI elements. Callbacks only set flags."""
        gui = self._server.gui

        # --- Clip Info ---
        with gui.add_folder("Clip Info", order=0):
            self._info_html = gui.add_html("<em>Loading...</em>")

        # --- Playback ---
        with gui.add_folder("Playback", order=1):
            self._play_btn = gui.add_button("Play", color="green")
            self._frame_slider = gui.add_slider(
                "Frame", min=0, max=1, step=1, initial_value=0,
            )
            self._speed_group = gui.add_button_group(
                "Speed", options=["0.25x", "0.5x", "1x", "2x"],
            )
            self._restart_btn = gui.add_button("Restart")

            @self._play_btn.on_click
            def _(_) -> None:
                self._pending_actions.append("toggle_play")

            @self._frame_slider.on_update
            def _(_) -> None:
                self._pending_frame_scrub = int(self._frame_slider.value)

            @self._speed_group.on_click
            def _(event) -> None:
                speed_map = {"0.25x": 0.25, "0.5x": 0.5, "1x": 1.0, "2x": 2.0}
                self._speed = speed_map.get(event.target.value, 1.0)

            @self._restart_btn.on_click
            def _(_) -> None:
                self._pending_actions.append("restart")

        # --- Annotation ---
        with gui.add_folder("Annotation", order=2):
            self._decision_dropdown = gui.add_dropdown(
                "Decision",
                options=["", "Keep", "Drop", "Skip"],
                initial_value="",
                hint="Must select Keep/Drop/Skip before saving",
            )
            self._difficulty_dropdown = gui.add_dropdown(
                "Difficulty",
                options=["", "easy", "medium", "hard", "bad_data"],
                initial_value="",
            )
            self._tags_input = gui.add_text("Issue Tags", initial_value="")
            self._note_input = gui.add_text("Note", initial_value="")
            self._save_next_btn = gui.add_button("Save & Next", color="blue")
            self._save_btn = gui.add_button("Save")

            @self._save_next_btn.on_click
            def _(_) -> None:
                self._pending_actions.append("save_next")

            @self._save_btn.on_click
            def _(_) -> None:
                self._pending_actions.append("save")

        # --- Navigation ---
        with gui.add_folder("Navigation", order=3):
            self._prev_btn = gui.add_button("Prev")
            self._next_btn = gui.add_button("Next")
            self._next_unreviewed_btn = gui.add_button("Next Unreviewed", color="orange")
            self._jump_input = gui.add_number(
                "Jump to #", initial_value=1, min=1, max=self._session.total, step=1,
            )

            @self._prev_btn.on_click
            def _(_) -> None:
                self._pending_actions.append("prev")

            @self._next_btn.on_click
            def _(_) -> None:
                self._pending_actions.append("next")

            @self._next_unreviewed_btn.on_click
            def _(_) -> None:
                self._pending_actions.append("next_unreviewed")

            @self._jump_input.on_update
            def _(_) -> None:
                self._pending_jump = int(self._jump_input.value)

        # --- Stats ---
        with gui.add_folder("Stats", order=4, expand_by_default=False):
            self._stats_html = gui.add_html("<em>Loading...</em>")

        # Visualization options
        self._scene.create_visualization_gui(show_debug_viz_control=False)

    # ------------------------------------------------------------------
    # Actions executed from the main loop (shallow call stack)
    # ------------------------------------------------------------------

    def _do_save(self) -> bool:
        """Save the current annotation. Returns False if no decision selected."""
        decision = self._decision_dropdown.value.lower() if self._decision_dropdown.value else ""
        if decision not in ("keep", "drop", "skip"):
            print("[REVIEW] WARNING: no decision selected, save skipped")
            return False
        self._session.annotate(
            decision=decision,
            difficulty=self._difficulty_dropdown.value,
            issue_tags=self._tags_input.value,
            note=self._note_input.value,
        )
        self._update_stats_display()

        # Terminal summary
        s = self._session.stats()
        row = self._session.current_row()
        clip_id = row.clip_id if row else "?"
        print(
            f"[REVIEW] {clip_id} -> {decision} | "
            f"{s.reviewed}/{s.total} ({s.progress_pct:.1f}%) | "
            f"keep={s.keep_count} drop={s.drop_count} skip={s.skip_count} | "
            f"kept_dur={s.kept_duration_s / 60:.1f}min"
        )
        return True

    def _load_current_clip(self) -> None:
        """Load the clip for the current session cursor."""
        row = self._session.current_row()
        if row is None:
            self._info_html.content = "<em>No clips to review</em>"
            return

        # Resolve NPZ path
        npz_path = Path(row.file_rel)
        if not npz_path.is_absolute():
            npz_path = self._project_root / npz_path

        try:
            self._player.load_clip(npz_path)
        except Exception as exc:
            self._info_html.content = f"<strong style='color:red'>Error loading clip:</strong><br/>{exc}"
            return

        self._current_frame = 0
        self._playing = False

        # Update scene first (before touching GUI widgets)
        self._player.set_frame(0)
        self._scene.update_from_mjdata(self._player.data)

        # Now update GUI widgets — disable slider callback processing
        # by setting _pending_frame_scrub to None after we're done.
        self._play_btn.label = "Play"
        self._play_btn.color = "green"

        max_frame = max(0, self._player.num_frames - 1)
        self._frame_slider.max = max_frame
        self._frame_slider.value = 0

        self._decision_dropdown.value = row.decision.capitalize() if row.decision else ""
        self._difficulty_dropdown.value = row.difficulty
        self._tags_input.value = row.issue_tags
        self._note_input.value = row.note
        self._jump_input.value = self._session.cursor_display

        # Drain any spurious pending events triggered by the GUI updates above
        self._pending_frame_scrub = None
        self._pending_jump = None
        self._pending_actions.clear()

        self._update_info_display()
        self._update_stats_display()

    def _update_frame(self) -> None:
        """Set frame on player and update the 3D scene."""
        self._player.set_frame(self._current_frame)
        self._scene.update_from_mjdata(self._player.data)

    def _update_info_display(self) -> None:
        """Update the Clip Info HTML panel."""
        row = self._session.current_row()
        if row is None:
            return

        status_color = {
            "keep": "green", "drop": "red", "skip": "orange", "": "gray",
        }
        status_label = row.decision if row.decision else "unreviewed"
        color = status_color.get(row.decision, "gray")

        # Metrics warnings
        metrics = self._session.get_metrics(row.clip_id)
        warnings_html = ""
        if metrics is not None:
            badges = []
            if metrics.first_frame_jump > 0.5:
                badges.append('<span style="color:red">[first_frame_jump]</span>')
            elif metrics.first_frame_jump > 0.2:
                badges.append('<span style="color:orange">[first_frame_jump]</span>')
            if metrics.root_height_max > 1.5:
                badges.append('<span style="color:red">[root_too_high]</span>')
            if metrics.root_height_min < 0.2:
                badges.append('<span style="color:red">[root_too_low]</span>')
            if metrics.body_lin_vel_max > 20.0:
                badges.append('<span style="color:red">[high_lin_vel]</span>')
            elif metrics.body_lin_vel_max > 10.0:
                badges.append('<span style="color:orange">[high_lin_vel]</span>')
            if metrics.body_ang_vel_max > 30.0:
                badges.append('<span style="color:orange">[high_ang_vel]</span>')
            if badges:
                warnings_html = "<br/><strong>Warnings:</strong> " + " ".join(badges)

            metrics_html = (
                f"<br/><strong>Metrics:</strong> "
                f"ffj={metrics.first_frame_jump:.3f} "
                f"h=[{metrics.root_height_min:.2f},{metrics.root_height_max:.2f}] "
                f"rspd={metrics.root_speed_max:.1f} "
                f"lv={metrics.body_lin_vel_max:.1f} "
                f"av={metrics.body_ang_vel_max:.1f}"
            )
        else:
            metrics_html = ""

        self._info_html.content = f"""
        <div style="font-size: 0.85em; line-height: 1.4; padding: 0.5em;">
          <strong>#{self._session.cursor_display}/{self._session.total}</strong>
          <span style="color:{color}; margin-left:0.5em">[{status_label}]</span><br/>
          <strong>clip_id:</strong> {row.clip_id}<br/>
          <strong>source:</strong> {row.source}<br/>
          <strong>split:</strong> {row.resolved_split}<br/>
          <strong>frames:</strong> {row.num_frames} | <strong>fps:</strong> {row.fps}
          | <strong>duration:</strong> {row.duration_s:.2f}s{warnings_html}{metrics_html}
        </div>
        """

    def _update_stats_display(self) -> None:
        """Update the Stats HTML panel."""
        s = self._session.stats()
        by_source_lines = "".join(
            f"<strong>{src}:</strong> {dur / 60:.1f} min<br/>"
            for src, dur in sorted(s.kept_duration_by_source.items())
        )
        self._stats_html.content = f"""
        <div style="font-size: 0.85em; line-height: 1.4; padding: 0.5em;">
          <h4 style="margin:0 0 0.3em">Progress</h4>
          <strong>Reviewed:</strong> {s.reviewed} / {s.total} ({s.progress_pct:.1f}%)<br/>
          <strong>Keep:</strong> {s.keep_count} |
          <strong>Drop:</strong> {s.drop_count} |
          <strong>Skip:</strong> {s.skip_count}<br/>
          <h4 style="margin:0.5em 0 0.3em">Kept Duration</h4>
          <strong>Total:</strong> {s.kept_duration_s / 60:.1f} min<br/>
          <strong>Train:</strong> {s.kept_train_duration_s / 60:.1f} min<br/>
          <strong>Val:</strong> {s.kept_val_duration_s / 60:.1f} min<br/>
          <h4 style="margin:0.5em 0 0.3em">By Source</h4>
          {by_source_lines if by_source_lines else "<em>none yet</em>"}
        </div>
        """

    # ------------------------------------------------------------------
    # Main loop — processes pending actions from callbacks
    # ------------------------------------------------------------------

    def _process_pending(self) -> None:
        """Process all pending actions from GUI callbacks."""
        # Handle frame scrub from slider (take latest value only)
        scrub = self._pending_frame_scrub
        if scrub is not None and not self._playing:
            self._pending_frame_scrub = None
            self._current_frame = scrub
            self._update_frame()

        # Handle jump input
        jump = self._pending_jump
        if jump is not None:
            self._pending_jump = None
            self._session.jump_to(jump)
            self._load_current_clip()
            return  # load_current_clip clears remaining actions

        # Handle button actions (process one per tick to keep things responsive)
        while self._pending_actions:
            action = self._pending_actions.pop(0)

            if action == "toggle_play":
                self._playing = not self._playing
                self._play_btn.label = "Pause" if self._playing else "Play"
                self._play_btn.color = "red" if self._playing else "green"

            elif action == "restart":
                self._current_frame = 0
                self._update_frame()
                self._frame_slider.value = 0
                self._pending_frame_scrub = None  # drain spurious slider event

            elif action == "save":
                self._do_save()

            elif action == "save_next":
                if self._do_save():
                    self._session.go_next_unreviewed()
                    self._load_current_clip()
                    return  # load clears actions

            elif action == "prev":
                self._session.go_prev()
                self._load_current_clip()
                return

            elif action == "next":
                self._session.go_next()
                self._load_current_clip()
                return

            elif action == "next_unreviewed":
                self._session.go_next_unreviewed()
                self._load_current_clip()
                return

    def run(self) -> None:
        """Main loop: process callbacks and handle playback timing."""
        self.setup_gui()
        self._load_current_clip()

        print(f"\nReview viewer ready at http://localhost:{self._server.get_port()}")
        print("Press Ctrl+C to exit.\n")

        last_frame_time = time.time()
        try:
            while True:
                now = time.time()

                # Check if scene visualization settings changed
                if self._scene.needs_update:
                    self._scene.refresh_visualization()

                # Process GUI actions from callbacks
                self._process_pending()

                # Advance playback
                if self._playing and self._player.num_frames > 0:
                    dt = now - last_frame_time
                    frames_to_advance = dt * self._player.fps * self._speed
                    if frames_to_advance >= 1.0:
                        self._current_frame += int(frames_to_advance)
                        if self._current_frame >= self._player.num_frames:
                            self._current_frame = 0
                            self._playing = False
                            self._play_btn.label = "Play"
                            self._play_btn.color = "green"
                        self._update_frame()
                        self._frame_slider.value = self._current_frame
                        self._pending_frame_scrub = None  # drain spurious event
                        last_frame_time = now
                else:
                    last_frame_time = now

                time.sleep(1.0 / 60.0)

        except KeyboardInterrupt:
            print("\nShutting down...")
            self._server.stop()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Web-based dataset clip review tool")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument(
        "--review", type=str, default=None,
        help="Path to review_state.csv (default: data/datasets/review/{dataset}/review_state.csv)",
    )
    parser.add_argument("--xml", type=str, default=None, help="Robot XML path")
    parser.add_argument("--port", type=int, default=8012, help="Viser server port")
    parser.add_argument(
        "--sort", type=str, default="unreviewed_first",
        choices=["unreviewed_first", "source", "duration_desc", "suspicion_desc"],
        help="Initial clip sort order",
    )
    parser.add_argument(
        "--skip_metrics", action="store_true",
        help="Skip computing anomaly metrics (faster startup)",
    )
    parser.add_argument(
        "--metrics_workers", type=int, default=4,
        help="Number of workers for metrics computation",
    )
    args = parser.parse_args()

    if args.review:
        review_path = Path(args.review)
        if not review_path.is_absolute():
            review_path = (PROJECT_ROOT / review_path).resolve()
    else:
        review_path = PROJECT_ROOT / "data" / "datasets" / "review" / args.dataset / "review_state.csv"

    if not review_path.is_file():
        print(f"ERROR: review state not found: {review_path}", file=sys.stderr)
        print("Run init_review_manifest.py first.", file=sys.stderr)
        sys.exit(1)

    xml_path = Path(args.xml) if args.xml else DEFAULT_XML
    if not xml_path.is_file():
        print(f"ERROR: robot XML not found: {xml_path}", file=sys.stderr)
        sys.exit(1)

    # Pre-compute metrics
    metrics: dict[str, ClipMetrics] | None = None
    if not args.skip_metrics:
        rows = load_review_state(review_path)
        cache_path = review_path.parent / "metrics_cache.json"
        metrics = compute_all_metrics(
            rows, PROJECT_ROOT, cache_path=cache_path, num_workers=args.metrics_workers,
        )

    app = ReviewViewerApp(
        review_path=review_path,
        xml_path=xml_path,
        project_root=PROJECT_ROOT,
        port=args.port,
        sort_mode=args.sort,
        metrics=metrics,
    )
    app.run()


if __name__ == "__main__":
    main()
