from ...utils.cmk_diagnostic import make_diagnostic_payload


class CMK_ImageMetrics:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        "INT",
        "INT",
        "INT",
        "INT",
        "CMK_DIAGNOSTIC",
    )

    RETURN_NAMES = (
        "image",
        "width",
        "height",
        "batch_size",
        "channels",
        "diagnostic",
    )

    FUNCTION = "run"
    CATEGORY = "CMK/Toolbox/Diagnostics"

    def run(self, image):
        shape = getattr(image, "shape", None)

        if shape is None or len(shape) < 3:
            width = 0
            height = 0
            batch_size = 0
            channels = 0
        else:
            batch_size = int(shape[0]) if len(shape) > 0 else 1
            height = int(shape[1])
            width = int(shape[2])
            channels = int(shape[3]) if len(shape) > 3 else 0

        summary = (
            f"Width       : {width}\n"
            f"Height      : {height}\n"
            f"Batch Size  : {batch_size}\n"
            f"Channels    : {channels}"
        )

        diagnostic = make_diagnostic_payload(
            title="Image Metrics",
            node="CMK Image Metrics",
            previews=[image],
            summary=summary,
            details=summary,
            mode="Metrics",
            metadata={
                "width": width,
                "height": height,
                "batch_size": batch_size,
                "channels": channels,
            },
        )

        return (
            image,
            width,
            height,
            batch_size,
            channels,
            diagnostic,
        )


class CMK_ImageQuickMetrics:
    """Passive image diagnostics node.

    Passes the image through unchanged and shows the most useful tensor metrics
    directly in the node UI. Only the image is exposed as an output, so the node
    remains a quick inline diagnostic tool instead of requiring extra display
    nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "CMK/Toolbox/Diagnostics"
    OUTPUT_NODE = True

    def run(self, image):
        shape = getattr(image, "shape", None)
        dtype = str(getattr(image, "dtype", "unknown"))

        if shape is None or len(shape) < 3:
            width = 0
            height = 0
            pixels = 0
            batch_size = 0
            tensor_shape = "unknown"
        else:
            batch_size = int(shape[0]) if len(shape) > 0 else 1
            height = int(shape[1])
            width = int(shape[2])
            pixels = width * height
            tensor_shape = " × ".join(str(int(v)) for v in shape)

        metrics = {
            "width": str(width),
            "height": str(height),
            "pixels": str(pixels),
            "batch_size": str(batch_size),
            "tensor_shape": tensor_shape,
            "dtype": dtype,
        }

        metrics_text = [f"{key}: {value}" for key, value in metrics.items()]

        return {
            "ui": {
                # Keep the generic text payload for ComfyUI's native preview path.
                "text": metrics_text,
                # Also expose each metric as its own UI payload. ComfyUI forwards
                # UI values to onExecuted; single-value arrays are the most robust
                # format across frontend versions.
                "width": [metrics["width"]],
                "height": [metrics["height"]],
                "pixels": [metrics["pixels"]],
                "batch_size": [metrics["batch_size"]],
                "tensor_shape": [metrics["tensor_shape"]],
                "dtype": [metrics["dtype"]],
                "metrics": metrics,
            },
            "result": (image,),
        }


NODE_CLASS_MAPPINGS = {
    "CMK_ImageMetrics": CMK_ImageMetrics,
    "CMK_ImageQuickMetrics": CMK_ImageQuickMetrics,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CMK_ImageMetrics": "CMK Image Metrics",
    "CMK_ImageQuickMetrics": "CMK Image QuickMetrics",
}
