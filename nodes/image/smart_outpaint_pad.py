import torch
import torch.nn.functional as F

from ...utils.cmk_diagnostic import make_diagnostic_payload


class CMK_SmartOutpaintPad:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "outpaint_on": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOOLEAN", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("image", "mask", "outpaint_on", "diagnostic")

    FUNCTION = "run"
    CATEGORY = "CMK/Toolbox/Image"

    def _resize_image(self, image, width, height):
        x = image.permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
        return x.permute(0, 2, 3, 1)

    def _resize_mask(self, mask, width, height):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        x = mask.unsqueeze(1)
        x = F.interpolate(x, size=(height, width), mode="nearest")
        return x.squeeze(1)

    def _mask_preview(self, mask):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        mask = mask.clamp(0.0, 1.0)
        return mask.unsqueeze(-1).repeat(1, 1, 1, 3)

    def _diagnostic(
        self,
        *,
        image_preview,
        mask_preview,
        outpaint_on,
        src_w,
        src_h,
        out_w,
        out_h,
        target_w,
        target_h,
        pad_left,
        pad_right,
        pad_top,
        pad_bottom,
        mask_source="Input",
    ):
        src_ratio = float(src_w) / float(src_h) if int(src_h) else 0.0
        target_ratio = float(target_w) / float(target_h) if int(target_h) else 0.0

        summary = "\n".join([
            f"Status        : {'Enabled' if bool(outpaint_on) else 'Bypass'}",
            f"Input Size    : {int(src_w)} x {int(src_h)}",
            f"Target Size   : {int(target_w)} x {int(target_h)}",
            f"Output Size   : {int(out_w)} x {int(out_h)}",
            f"Input Ratio   : {src_ratio:.4f}",
            f"Target Ratio  : {target_ratio:.4f}",
            f"Pad Left      : {int(pad_left)}",
            f"Pad Right     : {int(pad_right)}",
            f"Pad Top       : {int(pad_top)}",
            f"Pad Bottom    : {int(pad_bottom)}",
            f"Mask          : {mask_source}",
        ])

        details = "\n".join([
            summary,
            "",
            "Mask",
            "Padded Area   : 1.0",
            "Original Area : preserved",
        ])

        warnings = []
        if not bool(outpaint_on):
            warnings.append("outpaint disabled")

        stages = [
            {"title":"01 Source","subtitle":f"{int(src_w)} × {int(src_h)}","image":image_preview},
            {"title":"02 Mask","subtitle":"Input","image":mask_preview},
        ] if bool(outpaint_on) else [
            {"title":"01 Source","subtitle":f"{int(src_w)} × {int(src_h)}","image":image_preview},
        ]

        return make_diagnostic_payload(
            title="Smart Outpaint Pad",
            node="CMK Smart Outpaint Pad",
            previews=[image_preview] if not bool(outpaint_on) else [image_preview, mask_preview],
            stages=stages,
            summary=summary,
            details=details,
            mode="Outpaint" if bool(outpaint_on) else "Bypass",
            metadata={
                "enabled": bool(outpaint_on),
                "mask_source": str(mask_source),
                "input_width": int(src_w),
                "input_height": int(src_h),
                "target_width": int(target_w),
                "target_height": int(target_h),
                "output_width": int(out_w),
                "output_height": int(out_h),
                "input_ratio": src_ratio,
                "target_ratio": target_ratio,
                "pad_left": int(pad_left),
                "pad_right": int(pad_right),
                "pad_top": int(pad_top),
                "pad_bottom": int(pad_bottom),
            },
            warnings=warnings,
        )

    def run(self, image, width, height, outpaint_on, mask=None):
        batch, src_h, src_w, channels = image.shape

        if mask is None:
            mask = torch.zeros((batch, src_h, src_w), dtype=image.dtype, device=image.device)
            mask_source = "Generated empty"
        else:
            mask_source = "Input"
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            mask = mask.to(device=image.device, dtype=image.dtype)

        pad_left = pad_right = pad_top = pad_bottom = 0

        if not outpaint_on:
            diagnostic = self._diagnostic(
                image_preview=image,
                mask_preview=self._mask_preview(mask),
                outpaint_on=outpaint_on,
                src_w=src_w,
                src_h=src_h,
                out_w=src_w,
                out_h=src_h,
                target_w=width,
                target_h=height,
                pad_left=pad_left,
                pad_right=pad_right,
                pad_top=pad_top,
                pad_bottom=pad_bottom,
                mask_source=mask_source,
            )
            return (image, mask, outpaint_on, diagnostic)

        src_ratio = src_w / src_h
        target_ratio = width / height

        if src_ratio < target_ratio:
            new_w = round(src_h * target_ratio)
            pad_total = max(0, new_w - src_w)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left

        elif src_ratio > target_ratio:
            new_h = round(src_w / target_ratio)
            pad_total = max(0, new_h - src_h)
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top

        img_nchw = image.permute(0, 3, 1, 2)
        img_padded = F.pad(
            img_nchw,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0.0,
        ).permute(0, 2, 3, 1)

        mask_bchw = mask.unsqueeze(1)
        mask_padded = F.pad(
            mask_bchw,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=1.0,
        ).squeeze(1)

        image_out = self._resize_image(img_padded, width, height)
        mask_out = self._resize_mask(mask_padded, width, height)

        diagnostic = self._diagnostic(
            image_preview=image_out,
            mask_preview=self._mask_preview(mask_out),
            outpaint_on=outpaint_on,
            src_w=src_w,
            src_h=src_h,
            out_w=width,
            out_h=height,
            target_w=width,
            target_h=height,
            pad_left=pad_left,
            pad_right=pad_right,
            pad_top=pad_top,
            pad_bottom=pad_bottom,
            mask_source=mask_source,
        )

        return (image_out, mask_out, outpaint_on, diagnostic)


NODE_CLASS_MAPPINGS = {
    "CMK_SmartOutpaintPad": CMK_SmartOutpaintPad,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CMK_SmartOutpaintPad": "CMK Smart Outpaint Pad",
}
