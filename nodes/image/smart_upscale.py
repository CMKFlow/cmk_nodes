import torch
import folder_paths
import comfy.utils
import comfy.model_management as model_management
import comfy_extras.chainner_models.model_loading as model_loading

from ...pipe.cmk_log_pipe import cmk_add_block
from ...utils.cmk_diagnostic import make_diagnostic_payload


class CMK_SmartUpscaler:
    @classmethod
    def INPUT_TYPES(cls):
        models = folder_paths.get_filename_list("upscale_models")

        return {
            "required": {
                "image": ("IMAGE",),
                "limit_4x_mp": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 100.0, "step": 0.5}),
                "limit_2x_mp": ("FLOAT", {"default": 8.0, "min": 0.5, "max": 100.0, "step": 0.5}),
                "model_4x": (models,),
                "model_2x": (models,),
            }
        }

    RETURN_TYPES = ("IMAGE", "CMK_LOG_PIPE", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("image", "log_pipe", "diagnostic")

    FUNCTION = "run"
    CATEGORY = "CMK/Toolbox/Image"

    TILE_SIZE = 768
    OVERLAP = 32

    def load_upscale_model(self, model_name):
        model_path = folder_paths.get_full_path_or_raise("upscale_models", model_name)
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)

        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})

        model = model_loading.load_state_dict(sd).eval()
        return model

    def upscale(self, image, model):
        device = model_management.get_torch_device()
        model.to(device)

        in_img = image.movedim(-1, -3).to(device)

        tile = self.TILE_SIZE
        while True:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                    in_img.shape[3],
                    in_img.shape[2],
                    tile_x=tile,
                    tile_y=tile,
                    overlap=self.OVERLAP,
                )

                pbar = comfy.utils.ProgressBar(steps)

                out = comfy.utils.tiled_scale(
                    in_img,
                    lambda a: model(a),
                    tile_x=tile,
                    tile_y=tile,
                    overlap=self.OVERLAP,
                    upscale_amount=model.scale,
                    pbar=pbar,
                )

                break

            except model_management.OOM_EXCEPTION:
                tile //= 2
                if tile < 128:
                    raise

        model.cpu()
        out = torch.clamp(out.movedim(-3, -1), min=0.0, max=1.0)
        return out

    @staticmethod
    def _megapixels(width: int, height: int) -> float:
        return float(width * height) / 1_000_000.0

    @staticmethod
    def _size_text(width: int, height: int) -> str:
        return f"{int(width)} × {int(height)}"

    def _make_log_pipe(self, *, reason: str, selected_model: str, input_width: int, input_height: int, output_width: int, output_height: int):
        log_lines = [
            "Mode   : Auto",
            f"Reason : {reason}",
            f"Input  : {self._size_text(input_width, input_height)}",
            f"Output : {self._size_text(output_width, output_height)}",
            f"Model  : {selected_model}",
        ]
        return cmk_add_block({"blocks": []}, "Smart Upscaler", 30, log_lines, True)

    def _make_summary(
        self,
        *,
        reason: str,
        selected_model: str,
        input_width: int,
        input_height: int,
        output_width: int,
        output_height: int,
    ):
        input_mp = self._megapixels(input_width, input_height)
        output_mp = self._megapixels(output_width, output_height)
        return (
            "Mode             : Auto\n"
            f"Reason           : {reason}\n"
            "\n"
            f"Input Size       : {self._size_text(input_width, input_height)}\n"
            f"Input MP         : {input_mp:.2f}\n"
            "\n"
            f"Output Size      : {self._size_text(output_width, output_height)}\n"
            f"Output MP        : {output_mp:.2f}\n"
            "\n"
            f"Model            : {selected_model}"
        )

    def _make_diagnostic(
        self,
        *,
        image,
        output_image,
        reason: str,
        selected_model: str,
        input_width: int,
        input_height: int,
        output_width: int,
        output_height: int,
        scale_factor: int,
        limit_4x_mp: float,
        limit_2x_mp: float,
        warning: str | None = None,
    ):
        summary = self._make_summary(
            reason=reason,
            selected_model=selected_model,
            input_width=input_width,
            input_height=input_height,
            output_width=output_width,
            output_height=output_height,
        )

        stages = [
            {"title": "01 Source", "subtitle": self._size_text(input_width, input_height), "image": image},
            {"title": "02 Final", "subtitle": self._size_text(output_width, output_height), "image": output_image},
        ]

        warnings = [warning] if warning else []
        input_mp = self._megapixels(input_width, input_height)
        output_mp = self._megapixels(output_width, output_height)

        return make_diagnostic_payload(
            title="Smart Upscaler",
            node="CMK Smart Upscaler",
            previews=[image, output_image],
            stages=stages,
            summary=summary,
            details=summary,
            mode="Auto",
            metadata={
                "mode": "Auto",
                "reason": reason,
                "input_width": int(input_width),
                "input_height": int(input_height),
                "input_megapixels": float(input_mp),
                "output_width": int(output_width),
                "output_height": int(output_height),
                "output_megapixels": float(output_mp),
                "scale_factor": int(scale_factor),
                "model": str(selected_model),
                "selected_model": str(selected_model),
                "limit_4x_mp": float(limit_4x_mp),
                "limit_2x_mp": float(limit_2x_mp),
            },
            warnings=warnings,
            metrics={
                "input_mp": float(input_mp),
                "output_mp": float(output_mp),
                "scale_factor": int(scale_factor),
            },
        )

    def run(self, image, limit_4x_mp, limit_2x_mp, model_4x, model_2x):
        input_height = int(image.shape[1])
        input_width = int(image.shape[2])
        input_mp = self._megapixels(input_width, input_height)

        if input_mp <= float(limit_4x_mp):
            scale_factor = 4
            selected_model = model_4x
            reason = f"Image ≤ {float(limit_4x_mp):.1f} MP"
        elif input_mp <= float(limit_2x_mp):
            scale_factor = 2
            selected_model = model_2x
            reason = f"Image ≤ {float(limit_2x_mp):.1f} MP"
        else:
            scale_factor = 1
            selected_model = "Passthrough"
            reason = f"Image > {float(limit_2x_mp):.1f} MP"

            output_width = input_width
            output_height = input_height
            diagnostic = self._make_diagnostic(
                image=image,
                output_image=image,
                reason=reason,
                selected_model=selected_model,
                input_width=input_width,
                input_height=input_height,
                output_width=output_width,
                output_height=output_height,
                scale_factor=scale_factor,
                limit_4x_mp=limit_4x_mp,
                limit_2x_mp=limit_2x_mp,
                warning="no upscale applied (image above 2x limit)",
            )
            log_pipe = self._make_log_pipe(
                reason=reason,
                selected_model=selected_model,
                input_width=input_width,
                input_height=input_height,
                output_width=output_width,
                output_height=output_height,
            )
            return (image, log_pipe, diagnostic)

        model = self.load_upscale_model(selected_model)
        upscaled = self.upscale(image, model)
        output_height = int(upscaled.shape[1])
        output_width = int(upscaled.shape[2])

        diagnostic = self._make_diagnostic(
            image=image,
            output_image=upscaled,
            reason=reason,
            selected_model=selected_model,
            input_width=input_width,
            input_height=input_height,
            output_width=output_width,
            output_height=output_height,
            scale_factor=scale_factor,
            limit_4x_mp=limit_4x_mp,
            limit_2x_mp=limit_2x_mp,
        )
        log_pipe = self._make_log_pipe(
            reason=reason,
            selected_model=selected_model,
            input_width=input_width,
            input_height=input_height,
            output_width=output_width,
            output_height=output_height,
        )
        return (upscaled, log_pipe, diagnostic)


class CMK_SmartUpscalerPipe(CMK_SmartUpscaler):
    """Pipe-native Smart Upscaler.

    Public contract:
        IMAGE + LOG -> IMAGE + LOG + diagnostic

    The incoming LOG is copied and extended. IMAGE is never stored in a
    process pipe and the standalone Smart Upscaler remains unchanged.
    """

    @classmethod
    def INPUT_TYPES(cls):
        models = folder_paths.get_filename_list("upscale_models")
        return {
            "required": {
                "IMAGE": ("IMAGE",),
                "LOG": ("CMK_LOG_PIPE",),
                "enable": ("BOOLEAN", {"default": True}),
                "limit_4x_mp": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 100.0, "step": 0.5}),
                "limit_2x_mp": ("FLOAT", {"default": 8.0, "min": 0.5, "max": 100.0, "step": 0.5}),
                "model_4x": (models,),
                "model_2x": (models,),
            }
        }

    RETURN_TYPES = ("IMAGE", "CMK_LOG_PIPE", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("IMAGE", "LOG", "diagnostic")
    FUNCTION = "run_pipe"
    CATEGORY = "CMK/Developer/Pipe/Execute"

    def check_lazy_status(self, IMAGE=None, LOG=None, **kwargs):
        needed = []
        if IMAGE is None:
            needed.append("IMAGE")
        if LOG is None:
            needed.append("LOG")
        return needed

    def run_pipe(self, IMAGE, LOG, enable, limit_4x_mp, limit_2x_mp, model_4x, model_2x):
        if not isinstance(LOG, dict):
            raise ValueError("CMK Smart Upscaler -Pipe-: LOG is missing or invalid")

        image = IMAGE
        input_height = int(image.shape[1])
        input_width = int(image.shape[2])
        input_mp = self._megapixels(input_width, input_height)

        if not bool(enable):
            reason = "Local ENABLE is OFF"
            selected_model = "Passthrough"
            diagnostic = self._make_diagnostic(
                image=image,
                output_image=image,
                reason=reason,
                selected_model=selected_model,
                input_width=input_width,
                input_height=input_height,
                output_width=input_width,
                output_height=input_height,
                scale_factor=1,
                limit_4x_mp=limit_4x_mp,
                limit_2x_mp=limit_2x_mp,
                warning="upscaler disabled",
            )
            result_log = cmk_add_block(
                LOG,
                "Smart Upscaler",
                95,
                [
                    "STATUS          : DISABLED",
                    "MODE            : PASSTHROUGH",
                    f"INPUT           : {self._size_text(input_width, input_height)}",
                    f"OUTPUT          : {self._size_text(input_width, input_height)}",
                ],
                True,
            )
            return (image, result_log, diagnostic)

        if input_mp <= float(limit_4x_mp):
            scale_factor = 4
            selected_model = model_4x
            reason = f"Image ≤ {float(limit_4x_mp):.1f} MP"
        elif input_mp <= float(limit_2x_mp):
            scale_factor = 2
            selected_model = model_2x
            reason = f"Image ≤ {float(limit_2x_mp):.1f} MP"
        else:
            scale_factor = 1
            selected_model = "Passthrough"
            reason = f"Image > {float(limit_2x_mp):.1f} MP"

            diagnostic = self._make_diagnostic(
                image=image,
                output_image=image,
                reason=reason,
                selected_model=selected_model,
                input_width=input_width,
                input_height=input_height,
                output_width=input_width,
                output_height=input_height,
                scale_factor=scale_factor,
                limit_4x_mp=limit_4x_mp,
                limit_2x_mp=limit_2x_mp,
                warning="no upscale applied (image above 2x limit)",
            )
            result_log = cmk_add_block(
                LOG,
                "Smart Upscaler",
                95,
                [
                    "STATUS          : EXECUTED",
                    "MODE            : PASSTHROUGH",
                    f"SELECTION REASON: {reason}",
                    f"INPUT           : {self._size_text(input_width, input_height)}",
                    f"OUTPUT          : {self._size_text(input_width, input_height)}",
                    "MODEL           : PASSTHROUGH",
                ],
                True,
            )
            return (image, result_log, diagnostic)

        model = self.load_upscale_model(selected_model)
        upscaled = self.upscale(image, model)
        output_height = int(upscaled.shape[1])
        output_width = int(upscaled.shape[2])

        diagnostic = self._make_diagnostic(
            image=image,
            output_image=upscaled,
            reason=reason,
            selected_model=selected_model,
            input_width=input_width,
            input_height=input_height,
            output_width=output_width,
            output_height=output_height,
            scale_factor=scale_factor,
            limit_4x_mp=limit_4x_mp,
            limit_2x_mp=limit_2x_mp,
        )
        result_log = cmk_add_block(
            LOG,
            "Smart Upscaler",
            95,
            [
                "STATUS          : EXECUTED",
                f"MODE            : {scale_factor}x",
                f"SELECTION REASON: {reason}",
                f"INPUT           : {self._size_text(input_width, input_height)}",
                f"OUTPUT          : {self._size_text(output_width, output_height)}",
                f"MODEL           : {selected_model}",
            ],
            True,
        )
        return (upscaled, result_log, diagnostic)


NODE_CLASS_MAPPINGS = {
    "CMK_SmartUpscaler": CMK_SmartUpscaler,
    "CMK_SmartUpscalerPipe": CMK_SmartUpscalerPipe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CMK_SmartUpscaler": "CMK Smart Upscaler",
    "CMK_SmartUpscalerPipe": "CMK Smart Upscaler -Pipe-",
}
