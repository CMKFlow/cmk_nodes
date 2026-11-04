from fractions import Fraction

from comfy_api.latest import io, Input


def _safe_str(value, default="—"):
    if value is None:
        return default
    try:
        return str(value)
    except Exception:
        return default


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _first_existing_attr(obj, names):
    """Return the first non-callable, non-None attribute value found on obj."""
    if obj is None:
        return None
    for name in names:
        try:
            if not hasattr(obj, name):
                continue
            value = getattr(obj, name)
            if callable(value):
                continue
            if value is not None:
                return value
        except Exception:
            continue
    return None


def _shape_frame_count(obj):
    """Best-effort frame count from tensor/list-like video containers."""
    if obj is None:
        return None

    # Tensor / ndarray style: video frames are normally dimension 0.
    try:
        shape = getattr(obj, "shape", None)
        if shape is not None and len(shape) >= 1:
            value = int(shape[0])
            if value > 0:
                return value
    except Exception:
        pass

    # Some wrappers keep the actual frame tensor/list in a nested attribute.
    for attr in ("images", "frames", "video", "tensor", "data", "samples"):
        try:
            nested = getattr(obj, attr, None)
        except Exception:
            nested = None
        if nested is None or nested is obj:
            continue
        value = _shape_frame_count(nested)
        if value is not None:
            return value

    # Plain list/tuple fallback.
    try:
        if isinstance(obj, (list, tuple)) and len(obj) > 0:
            return len(obj)
    except Exception:
        pass

    return None


def _video_metrics(video: Input.Video):
    """Collect robust video metrics without modifying the video object."""
    width = 0
    height = 0
    fps = 0
    bit_depth = 0
    frames_value = None

    try:
        width, height = video.get_dimensions()
        width = int(width)
        height = int(height)
    except Exception:
        width = 0
        height = 0

    components = None
    try:
        components = video.get_components()
    except Exception:
        components = None

    if components is not None:
        fps_value = _first_existing_attr(components, ("frame_rate", "fps", "framerate"))
        fps = _safe_float(fps_value, 0.0)

        frames_value = _shape_frame_count(components)
        if frames_value is None:
            frames_value = _first_existing_attr(
                components,
                ("frame_count", "num_frames", "n_frames", "total_frames", "frames_count"),
            )

    if frames_value is None:
        frames_value = _shape_frame_count(video)

    try:
        bit_depth = int(video.get_bit_depth())
    except Exception:
        bit_depth = 0

    frames_int = None
    try:
        if frames_value is not None:
            frames_int = int(frames_value)
    except Exception:
        frames_int = None

    if frames_int is not None and frames_int > 0:
        frames = str(frames_int)
    else:
        frames = "—"

    if frames_int is not None and frames_int > 0 and fps not in (None, 0, 0.0):
        duration_seconds = frames_int / float(fps)
        duration = f"{duration_seconds:.3f} s"
    else:
        duration = "—"

    fps_display = str(int(round(fps))) if fps else "0"
    pixels_per_frame = width * height if width and height else 0

    return {
        "width": str(width),
        "height": str(height),
        "pixels_per_frame": str(pixels_per_frame),
        "fps": fps_display,
        "bit_depth": str(bit_depth),
        "frames": frames,
        "duration": duration,
    }


class CMK_VideoMetrics(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CMK_VideoMetrics",
            display_name="CMK Video Metrics",
            category="CMK/Toolbox/Diagnostics",
            inputs=[
                io.Video.Input("video"),
            ],
            outputs=[
                io.Video.Output("video"),
                io.Int.Output("width"),
                io.Int.Output("height"),
                io.Int.Output("fps"),
                io.Int.Output("bit_depth"),
            ],
        )

    @classmethod
    def execute(cls, video: Input.Video) -> io.NodeOutput:
        metrics = _video_metrics(video)

        return io.NodeOutput(
            video,
            _safe_int(metrics["width"]),
            _safe_int(metrics["height"]),
            _safe_int(metrics["fps"]),
            _safe_int(metrics["bit_depth"]),
        )


class CMK_VideoQuickMetrics(io.ComfyNode):
    """Passive video diagnostics node with direct in-node metric display."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CMK_VideoQuickMetrics",
            display_name="CMK Video QuickMetrics",
            category="CMK/Toolbox/Diagnostics",
            inputs=[
                io.Video.Input("video"),
            ],
            outputs=[
                io.Video.Output("video"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, video: Input.Video) -> io.NodeOutput:
        metrics = _video_metrics(video)
        metrics_text = [f"{key}: {value}" for key, value in metrics.items()]

        return io.NodeOutput(
            video,
            ui={
                "text": metrics_text,
                "width": [metrics["width"]],
                "height": [metrics["height"]],
                "pixels_per_frame": [metrics["pixels_per_frame"]],
                "fps": [metrics["fps"]],
                "bit_depth": [metrics["bit_depth"]],
                "frames": [metrics["frames"]],
                "duration": [metrics["duration"]],
                "metrics": metrics,
            },
        )


NODE_CLASS_MAPPINGS = {
    "CMK_VideoMetrics": CMK_VideoMetrics,
    "CMK_VideoQuickMetrics": CMK_VideoQuickMetrics,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CMK_VideoMetrics": "CMK Video Metrics",
    "CMK_VideoQuickMetrics": "CMK Video QuickMetrics",
}
