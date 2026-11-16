#!/usr/bin/env python3
"""Interactive web-based 3D point-cloud viewer for monocular depth videos.

Visualizes per-frame relative depth produced by ``run_video_trt.py`` (Depth
Anything V2) as an animated, orbit-able 3D point cloud in the browser, using
viser. Unlike a global-reconstruction viewer (e.g. Endo3R's ``vis.py``), each
video frame is back-projected independently with a pinhole camera model and
streamed into a single point-cloud object.

Input .npz contract (written by run_video_trt.py):
    depths      (N, H, W) float16  raw DA-V2 relative depth (larger == closer,
                                   i.e. disparity-like; NOT normalized)
    fps         float              source video frame rate
    width       int                source video width
    height      int                source video height
    video_path  str                absolute path to the source RGB video
    flow_path   str (optional)     path to an optical-flow velocity-field
                                   visualization video for the side panel

Usage:
    python vis_video.py --depth output/clip.npz [--video rgb.mp4]
                        [--flow flow.mp4] [--port 8080]
"""

from __future__ import annotations

import argparse
import math
import os
import threading
import time
from collections import OrderedDict

import cv2
import numpy as np
import viser
from matplotlib import colormaps


# --------------------------------------------------------------------------- CLI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Web-based 3D point-cloud viewer for depth videos (viser).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--depth",
        required=True,
        help="Path to the .npz written by run_video_trt.py",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Optional override for the RGB video path stored in the npz",
    )
    parser.add_argument(
        "--flow",
        default=None,
        help="Optional path to an optical-flow velocity-field visualization "
        "video (falls back to the npz 'flow_path' entry)",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--fov-deg",
        type=float,
        default=55.0,
        help="Horizontal pinhole camera FOV in degrees",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=250_000,
        help="Point budget; an integer grid stride is chosen to stay under it",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Explicit subsampling stride (overrides --max-points)",
    )
    return parser.parse_args()


# ----------------------------------------------------------------- video source


class _VideoFrameSource:
    """``cv2.VideoCapture`` wrapper with a sequential-read fast path and a
    soft-capped FIFO frame cache.

    Scrubs via ``CAP_PROP_POS_FRAMES``; during normal playback frames are
    decoded with plain sequential ``read()`` calls. Cached frames are kept at
    native resolution in an insertion-ordered dict; once the byte budget is
    exceeded the oldest frames are evicted (they can always be re-decoded).
    The clips handled here are short, so in practice everything fits.
    """

    def __init__(self, path: str, cache_cap_bytes: int = 256 * 1024 * 1024):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.ok = bool(self.cap.isOpened())
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_bytes = 0
        self._cache_cap_bytes = cache_cap_bytes
        self._next_idx: int | None = None  # frame index the cap will read next

    def get(self, idx: int) -> np.ndarray | None:
        """Return the RGB uint8 frame at ``idx`` (None on read failure)."""
        if not self.ok:
            return None
        hit = self._cache.get(idx)
        if hit is not None:
            self._cache.move_to_end(idx)
            return hit
        if self._next_idx != idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = self.cap.read()
        if not ok:
            self._next_idx = None
            return None
        self._next_idx = idx + 1
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._cache[idx] = rgb
        self._cache_bytes += rgb.nbytes
        while self._cache_bytes > self._cache_cap_bytes and len(self._cache) > 1:
            _, old = self._cache.popitem(last=False)
            self._cache_bytes -= old.nbytes
        return rgb


# ------------------------------------------------------------------------ viewer


class DepthVideoViewer:
    """Streams per-frame back-projected depth into one viser point cloud."""

    COLOR_RGB = "RGB (video)"
    COLOR_DEPTH = "Depth (turbo)"

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.port = args.port
        self.depth_path = args.depth

        # ------------------------------------------------------------- load npz
        data = np.load(args.depth)
        if "depths" not in data.files:
            raise KeyError(f"{args.depth} does not contain a 'depths' array")
        depths = data["depths"]
        if depths.ndim != 3:
            raise ValueError(f"'depths' should be (N, H, W), got {depths.shape}")
        self.depths = depths
        self.n_frames, self.h, self.w = depths.shape

        fps = float(data["fps"]) if "fps" in data.files else 30.0
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        self.video_fps = fps

        video_path = args.video
        if video_path is None and "video_path" in data.files:
            video_path = str(data["video_path"])
        self.video: _VideoFrameSource | None = None
        if video_path:
            if not os.path.isfile(video_path):
                print(f"  [warn] RGB video not found: {video_path} "
                      "(falling back to depth colors)")
            else:
                self.video = _VideoFrameSource(video_path)
                if not self.video.ok:
                    print(f"  [warn] could not open RGB video: {video_path}")

        flow_path = args.flow
        if flow_path is None and "flow_path" in data.files:
            flow_path = str(data["flow_path"])
        self.flow_video: _VideoFrameSource | None = None
        if flow_path:
            if not os.path.isfile(flow_path):
                print(f"  [warn] flow video not found: {flow_path}")
            else:
                self.flow_video = _VideoFrameSource(flow_path)
                if not self.flow_video.ok:
                    print(f"  [warn] could not open flow video: {flow_path}")

        # ------------------------------------------------------- subsample grid
        if args.stride is not None:
            stride = max(1, int(args.stride))
        else:
            stride = max(
                1,
                int(math.ceil(math.sqrt((self.h * self.w) / max(1, args.max_points)))),
            )
        self.stride = stride
        vs = np.arange(0, self.h, stride)
        us = np.arange(0, self.w, stride)
        uu, vv = np.meshgrid(us, vs)  # (Gh, Gw)
        self.grid_w = uu.shape[1]
        self.n_points = int(uu.size)

        # --------------------------------------------- pinhole intrinsics (fov)
        # fx = fy = 0.5 * W / tan(fov / 2);  cx = W / 2, cy = H / 2
        f = 0.5 * self.w / math.tan(math.radians(args.fov_deg) / 2.0)
        self.fx = self.fy = f
        self.cx, self.cy = self.w / 2.0, self.h / 2.0
        # Per-pixel normalized rays; y is flipped so that +y is "up" in the
        # world and the image appears upright.
        self.ray_x = ((uu - self.cx) / self.fx).astype(np.float32)
        self.ray_y = (-(vv - self.cy) / self.fy).astype(np.float32)

        # --------------------------- global robust normalization (computed ONCE
        # over all frames so the z mapping does not flicker during playback)
        n_sample = min(self.n_frames, 48)
        sample_frames = np.linspace(0, self.n_frames - 1, n_sample).astype(np.int64)
        pix_step = max(1, int(math.sqrt(self.h * self.w / 20_000)))
        sample = (
            depths[sample_frames][:, ::pix_step, ::pix_step]
            .astype(np.float32)
            .ravel()
        )
        lo, hi = np.percentile(sample, [1.0, 99.0])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
            lo = float(np.min(sample))
            hi = float(max(np.max(sample), lo + 1e-6))
        self.disp_lo, self.disp_hi = float(lo), float(hi)

        # Turbo LUT for the depth-color mode.
        self.depth_lut = (
            colormaps["turbo"](np.linspace(0.0, 1.0, 256))[:, :3] * 255.0
        ).astype(np.uint8)

        # ------------------------------------------------- depth mapping state
        self.z_near = 0.5
        self.z_far = 10.0
        tan_h = math.tan(math.radians(args.fov_deg) / 2.0)
        self.vfov_rad = 2.0 * math.atan(tan_h * self.h / self.w)
        # Half-width of the cloud at the far plane: a convenient scene radius
        # used for framing, grid sizing and the default point size.
        self.scene_r = self.z_far * tan_h

        # ------------------------------------------------------------ playback
        self._frame = 0
        self._playing = self.n_frames > 1
        self._last_advance = time.time()
        self._remap_guard = False
        self._lock = threading.Lock()
        self.color_mode = self.COLOR_RGB if self.video is not None else self.COLOR_DEPTH

        # --------------------------------------------------------------- server
        self.server = viser.ViserServer(port=args.port)
        self.server.gui.configure_theme(
            dark_mode=True,
            control_width="medium",
            show_logo=False,
            show_share_button=False,
        )
        self.server.scene.set_up_direction("+y")

        # Default orbit view: in front of the cloud, slightly above, looking
        # at the (centered) scene. Also becomes the "Reset View" target.
        r = self.scene_r
        self.server.initial_camera.position = (0.0, 0.45 * r, -1.55 * r)
        self.server.initial_camera.look_at = (0.0, 0.0, 0.0)

        # ------------------------------------------------------------ scene nodes
        y_bottom = -self.z_far * math.tan(self.vfov_rad / 2.0)
        self.grid = self.server.scene.add_grid(
            "/grid",
            width=3.2 * r,
            height=3.2 * r,
            plane="xz",
            position=(0.0, y_bottom - 0.12 * r, 0.0),
            cell_color=(70, 74, 88),
            section_color=(118, 124, 146),
            cell_size=r / 4.0,
            section_size=r,
            shadow_opacity=0.0,
        )
        # Virtual capture camera: sits at the origin of the camera frame,
        # i.e. at z = -z_mid in the centered world frame.
        self.cam_axes = self.server.scene.add_frame(
            "/capture_camera/axes",
            axes_length=0.28 * r,
            axes_radius=0.012 * r,
            position=(0.0, 0.0, -self._z_mid()),
        )
        self.frustum = self.server.scene.add_camera_frustum(
            "/capture_camera/frustum",
            fov=self.vfov_rad,
            aspect=self.w / self.h,
            scale=self._frustum_scale(),
            color=(56, 189, 248),
            # 180 deg about the optical axis: viser frustums follow the OpenCV
            # convention (+Y down) while our world is +Y up.
            wxyz=(0.0, 0.0, 0.0, 1.0),
            position=(0.0, 0.0, -self._z_mid()),
            variant="wireframe",
        )
        self.cloud = self.server.scene.add_point_cloud(
            "/point_cloud",
            points=self._frame_geometry(0),
            colors=self._frame_colors(0, rgb_frame=self._get_rgb(0)),
            point_size=self._default_point_size(),
            point_shape="circle",
            precision="float16",
        )

        # ----------------------------------------------------------------- GUI
        self._build_gui()

        # ------------------------------------------------------ playback thread
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------- helpers

    def _z_mid(self) -> float:
        return 0.5 * (self.z_near + self.z_far)

    def _frustum_scale(self) -> float:
        return max(0.5 * self.z_far, 0.3)

    def _default_point_size(self) -> float:
        return float(np.clip(1.2 * 2.0 * self.scene_r / self.grid_w, 0.001, 0.1))

    def _get_rgb(self, idx: int) -> np.ndarray | None:
        """RGB frame resized to the depth resolution (for colors + panel)."""
        if self.video is None or not self.video.ok:
            return None
        rgb = self.video.get(idx)
        if rgb is not None and rgb.shape[:2] != (self.h, self.w):
            rgb = cv2.resize(rgb, (self.w, self.h), interpolation=cv2.INTER_AREA)
        return rgb

    def _get_flow(self, idx: int) -> np.ndarray | None:
        """Optical-flow visualization frame resized to the depth resolution."""
        if self.flow_video is None or not self.flow_video.ok:
            return None
        flow = self.flow_video.get(idx)
        if flow is not None and flow.shape[:2] != (self.h, self.w):
            flow = cv2.resize(flow, (self.w, self.h), interpolation=cv2.INTER_AREA)
        return flow

    def _disp_norm(self, frame_idx: int) -> np.ndarray:
        """Raw disparity-like values -> t in [0, 1] using the GLOBAL robust
        percentiles (1st / 99th) computed once at startup."""
        raw = self.depths[frame_idx][:: self.stride, :: self.stride].astype(np.float32)
        t = (raw - self.disp_lo) / (self.disp_hi - self.disp_lo)
        return np.clip(t, 0.0, 1.0)

    def _frame_geometry(self, frame_idx: int) -> np.ndarray:
        """Back-project one frame to a centered (M, 3) point cloud."""
        t = self._disp_norm(frame_idx)
        # DA-V2 output is disparity-like: larger raw value -> closer point.
        z = self.z_far - t * (self.z_far - self.z_near)
        x = self.ray_x * z
        y = self.ray_y * z
        # Center the cloud on the origin so viser's default orbit camera
        # (which looks at the origin) frames it nicely.
        z_world = z - self._z_mid()
        return np.stack([x, y, z_world], axis=-1).reshape(-1, 3).astype(np.float32)

    def _frame_colors(
        self, frame_idx: int, rgb_frame: np.ndarray | None = None
    ) -> np.ndarray:
        if self.color_mode == self.COLOR_RGB:
            if rgb_frame is None:
                rgb_frame = self._get_rgb(frame_idx)
            if rgb_frame is not None:
                return rgb_frame[:: self.stride, :: self.stride].reshape(-1, 3)
        # Depth colormap of z. t == 1 - (z - near) / (far - near), so the LUT
        # on t gives the conventional near == warm / far == cool mapping.
        t = self._disp_norm(frame_idx)
        return self.depth_lut[(t * 255.0).astype(np.uint8)].reshape(-1, 3)

    # ------------------------------------------------------------------- GUI

    def _build_gui(self) -> None:
        gui = self.server.gui

        gui.add_markdown(
            f"## Depth video viewer\n"
            f"`{os.path.basename(self.depth_path)}` — {self.n_frames} frames · "
            f"{self.w}×{self.h} · {self.video_fps:g} fps · "
            f"{self.n_points:,} pts (stride {self.stride})",
            order=0,
        )

        # Videos live in a standalone panel docked to the LEFT edge; all
        # controls stay in the main control panel, docked to the RIGHT.
        self.image_handle = None
        self.flow_image_handle = None
        has_rgb = self.video is not None and self.video.ok
        has_flow = self.flow_video is not None and self.flow_video.ok
        if has_rgb or has_flow:
            video_panel = gui.add_panel()
            with video_panel.add_tab("Videos", viser.Icon.VIDEO):
                if has_rgb:
                    first = self._get_rgb(0)
                    if first is None:
                        first = np.zeros((self.h, self.w, 3), dtype=np.uint8)
                    self.image_handle = gui.add_image(
                        first,
                        label="RGB video" if has_flow else None,
                        format="jpeg",
                        jpeg_quality=80,
                    )
                if has_flow:
                    first_flow = self._get_flow(0)
                    if first_flow is None:
                        first_flow = np.zeros((self.h, self.w, 3), dtype=np.uint8)
                    self.flow_image_handle = gui.add_image(
                        first_flow,
                        label="Optical flow",
                        format="jpeg",
                        jpeg_quality=80,
                    )
            video_panel.dock_left()
            # Wide enough for a 16:9-ish frame; the user can resize freely.
            video_panel.set_width(480)
        gui.main_panel.dock_right()

        with gui.add_folder("Playback", order=2):
            self.play_button = gui.add_button(
                "Pause" if self._playing else "Play",
                icon=viser.Icon.PLAYER_PAUSE if self._playing else viser.Icon.PLAYER_PLAY,
                hint="Play / pause the depth video",
            )
            self.frame_slider = gui.add_slider(
                "Frame",
                min=0,
                max=self.n_frames - 1,
                step=1,
                initial_value=0,
                marks=((0, "0"), (self.n_frames - 1, f"{self.n_frames - 1}"))
                if self.n_frames > 1
                else None,
                hint="Scrub through the video",
            )
            self.fps_slider = gui.add_slider(
                "FPS",
                min=0.5,
                max=60.0,
                step=0.5,
                initial_value=float(min(60.0, max(0.5, round(self.video_fps * 2) / 2))),
            )
            self.loop_checkbox = gui.add_checkbox("Loop", True)

        with gui.add_folder("Depth mapping", order=3):
            self.near_slider = gui.add_slider(
                "Near plane (z)",
                min=0.05,
                max=20.0,
                step=0.05,
                initial_value=self.z_near,
                hint="Closest depth-plane distance; remaps z live",
            )
            self.far_slider = gui.add_slider(
                "Far plane (z)",
                min=0.5,
                max=60.0,
                step=0.5,
                initial_value=self.z_far,
                hint="Farthest depth-plane distance; remaps z live",
            )

        with gui.add_folder("Appearance", order=4):
            self.point_size_slider = gui.add_slider(
                "Point size",
                min=0.001,
                max=0.1,
                step=0.001,
                initial_value=self._default_point_size(),
            )
            color_options = [self.COLOR_DEPTH]
            if self.video is not None and self.video.ok:
                color_options.insert(0, self.COLOR_RGB)
            self.color_dropdown = gui.add_dropdown(
                "Point colors",
                color_options,
                initial_value=self.color_mode,
            )
            self.frustum_checkbox = gui.add_checkbox("Camera frustum", True)
            self.grid_checkbox = gui.add_checkbox("Grid", True)

        gui.add_markdown(
            "---\n*drag to orbit · scroll to zoom · right-drag to pan*", order=99
        )

        # --------------------------------------------------------- callbacks

        @self.play_button.on_click
        def _(_) -> None:
            if self._playing:
                self._set_playing(False)
            else:
                # Restart from the top when resuming at the last frame.
                if self._frame >= self.n_frames - 1:
                    self.show_frame(0)
                self._set_playing(True)

        @self.frame_slider.on_update
        def _(event: viser.GuiEvent) -> None:
            # Server-side syncs (playback loop) arrive with client=None and
            # are ignored; only genuine user drags scrub.
            if event.client is None:
                return
            self.show_frame(int(self.frame_slider.value), sync_slider=False)

        @self.near_slider.on_update
        def _(event: viser.GuiEvent) -> None:
            self._on_remap(event)

        @self.far_slider.on_update
        def _(event: viser.GuiEvent) -> None:
            self._on_remap(event)

        @self.point_size_slider.on_update
        def _(_) -> None:
            self.cloud.point_size = float(self.point_size_slider.value)

        @self.color_dropdown.on_update
        def _(_) -> None:
            self.color_mode = str(self.color_dropdown.value)
            with self._lock:
                self.cloud.colors = self._frame_colors(self._frame)

        @self.frustum_checkbox.on_update
        def _(_) -> None:
            visible = bool(self.frustum_checkbox.value)
            self.frustum.visible = visible
            self.cam_axes.visible = visible

        @self.grid_checkbox.on_update
        def _(_) -> None:
            self.grid.visible = bool(self.grid_checkbox.value)

    # -------------------------------------------------------------- behaviour

    def _set_playing(self, flag: bool) -> None:
        self._playing = flag
        self.play_button.label = "Pause" if flag else "Play"
        self.play_button.icon = (
            viser.Icon.PLAYER_PAUSE if flag else viser.Icon.PLAYER_PLAY
        )
        self._last_advance = time.time()

    def _on_remap(self, event: viser.GuiEvent) -> None:
        """Near/far sliders moved: keep near < far, then remap z live."""
        if self._remap_guard:
            return
        self._remap_guard = True
        try:
            near = float(self.near_slider.value)
            far = float(self.far_slider.value)
            if near >= far:
                if event.target is self.near_slider:
                    far = near + 0.05
                    self.far_slider.value = far
                else:
                    near = max(0.05, far - 0.05)
                    self.near_slider.value = near
            self.z_near, self.z_far = near, far
            with self._lock:
                self.cloud.points = self._frame_geometry(self._frame)
                self.frustum.position = (0.0, 0.0, -self._z_mid())
                self.frustum.scale = self._frustum_scale()
                self.cam_axes.position = (0.0, 0.0, -self._z_mid())
        finally:
            self._remap_guard = False

    def show_frame(self, idx: int, sync_slider: bool = True) -> None:
        """Render one frame: point cloud + video panels (+ slider position)."""
        idx = int(np.clip(idx, 0, self.n_frames - 1))
        with self._lock:
            self._frame = idx
            rgb = self._get_rgb(idx)
            self.cloud.points = self._frame_geometry(idx)
            self.cloud.colors = self._frame_colors(idx, rgb_frame=rgb)
            if rgb is not None and self.image_handle is not None:
                self.image_handle.image = rgb
            flow = self._get_flow(idx)
            if flow is not None and self.flow_image_handle is not None:
                self.flow_image_handle.image = flow
            if sync_slider:
                # Fires the slider callback with client=None, which is ignored.
                self.frame_slider.value = idx
            self._last_advance = time.time()

    def _playback_loop(self) -> None:
        """Daemon thread: advance the frame at 1 / fps while playing."""
        while True:
            if not self._playing:
                time.sleep(0.05)
                continue
            fps = max(1e-3, float(self.fps_slider.value))
            period = 1.0 / fps
            wait = period - (time.time() - self._last_advance)
            if wait > 0:
                time.sleep(min(wait, 0.1))
                continue
            nxt = self._frame + 1
            if nxt >= self.n_frames:
                if bool(self.loop_checkbox.value):
                    nxt = 0
                else:
                    self._set_playing(False)
                    continue
            self.show_frame(nxt)

    # -------------------------------------------------------------------- run

    def run(self) -> None:
        url = f"http://localhost:{self.port}"
        flow_note = (
            " | flow video"
            if self.flow_video is not None and self.flow_video.ok
            else ""
        )
        print(
            f"\n  Depth video viewer running at {url}\n"
            f"  {self.n_frames} frames | {self.w}x{self.h} | "
            f"{self.n_points:,} points/frame (stride {self.stride})"
            f"{flow_note}\n",
            flush=True,
        )
        try:
            while True:
                time.sleep(10.0)
        except KeyboardInterrupt:
            pass
        finally:
            self.server.stop()


def main() -> None:
    args = parse_args()
    viewer = DepthVideoViewer(args)
    viewer.run()


if __name__ == "__main__":
    main()
