from ..cmk_common import RESOLVE_LATENT_SOURCES, RESOLVE_IMAGE_SOURCES, first_not_none, optional_bool


class CMKSmartPipeResolveState:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("CMK_PIPE",),
                "image_source": (RESOLVE_IMAGE_SOURCES,),
                "latent_source": (RESOLVE_LATENT_SOURCES,),
            }
        }

    RETURN_TYPES = ("CMK_PIPE", "IMAGE", "LATENT", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("pipe", "image", "latent_image", "used_vae_encode", "resolve_image_log", "resolve_latent_log")
    FUNCTION = "resolve_state"
    CATEGORY = "CMK/Developer/Legacy"

    def resolve_state(self, pipe, image_source, latent_source):
        image_current = pipe.get("image")
        image_1st_pass = pipe.get("image_1st_pass")
        image_original = pipe.get("image_original")
        latent_current = pipe.get("latent_image")
        latent_1st_pass = pipe.get("latent_1st_pass")
        latent_original = pipe.get("latent_original")
        vae = pipe.get("vae")

        detailer_active = bool(pipe.get("boolean_detailer_enable", False))
        face_active = bool(pipe.get("boolean_face_enable", False))
        upstream_image_changed = detailer_active or face_active

        if image_source == "Original":
            image_out = first_not_none(image_original, image_1st_pass, image_current)
            image_log = "image_source=Original | original image"
        elif image_source == "Current" or image_source == "Processed":
            image_out = first_not_none(image_current, image_1st_pass, image_original)
            image_log = f"image_source={image_source} | current pipe image"
        else:
            image_out = first_not_none(image_current, image_1st_pass, image_original)
            image_log = "image_source=Auto | current pipe image"

        if latent_source == "Original":
            needs_encode = False
            latent_base = first_not_none(latent_1st_pass, latent_original, latent_current)
            latent_decision_log = "latent_source=Original | original latent"
        elif latent_source == "Encoded":
            needs_encode = True
            latent_base = first_not_none(latent_1st_pass, latent_original, latent_current)
            latent_decision_log = "latent_source=Encoded | requested vae encode"
        else:
            needs_encode = upstream_image_changed
            latent_base = first_not_none(latent_1st_pass, latent_original, latent_current)
            if needs_encode:
                latent_decision_log = "latent_source=Auto | upstream image changed -> vae encode"
            else:
                latent_decision_log = "latent_source=Auto | no upstream image change -> original latent"

        if needs_encode:
            if image_out is None:
                return (pipe, image_out, latent_base, False, image_log, latent_decision_log + " | fallback: image missing")
            if vae is None:
                return (pipe, image_out, latent_base, False, image_log, latent_decision_log + " | fallback: vae missing")

            encoded = vae.encode(image_out)
            latent_out = {"samples": encoded}
            new_pipe = dict(pipe)
            new_pipe["image"] = image_out
            new_pipe["latent_image"] = latent_out
            return (new_pipe, image_out, latent_out, True, image_log, latent_decision_log)

        new_pipe = dict(pipe)
        new_pipe["image"] = image_out
        new_pipe["latent_image"] = latent_base
        return (new_pipe, image_out, latent_base, False, image_log, latent_decision_log)


class CMKSmartPipeResolveLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("CMK_PIPE",),
                "mode": (RESOLVE_LATENT_SOURCES,),
            },
            "optional": {
                "opt_detailer_enable": ("BOOLEAN", {"forceInput": True}),
                "opt_face_enable": ("BOOLEAN", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("CMK_PIPE", "LATENT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("pipe", "latent_image", "used_vae_encode", "resolve_latent_log")
    FUNCTION = "resolve_latent"
    CATEGORY = "CMK/Developer/Legacy"

    def resolve_latent(self, pipe, mode, opt_detailer_enable=None, opt_face_enable=None):
        image = pipe.get("image")
        latent_base = first_not_none(pipe.get("latent_1st_pass"), pipe.get("latent_original"), pipe.get("latent_image"))
        vae = pipe.get("vae")
        detailer_active = optional_bool(opt_detailer_enable, False)
        face_active = optional_bool(opt_face_enable, False)

        if mode == "Original":
            needs_encode = False
        elif mode == "Encoded":
            needs_encode = True
        else:
            needs_encode = detailer_active or face_active

        if needs_encode:
            if image is None:
                return (pipe, latent_base, False, f"{mode} | fallback: image missing")
            if vae is None:
                return (pipe, latent_base, False, f"{mode} | fallback: vae missing")

            latent_out = {"samples": vae.encode(image)}
            new_pipe = dict(pipe)
            new_pipe["latent_image"] = latent_out
            return (new_pipe, latent_out, True, f"{mode} | vae encode")

        new_pipe = dict(pipe)
        new_pipe["latent_image"] = latent_base
        return (new_pipe, latent_base, False, f"{mode} | original latent")


class CMKSmartPipeResolveImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("CMK_PIPE",),
                "mode": (RESOLVE_IMAGE_SOURCES,),
            }
        }

    RETURN_TYPES = ("CMK_PIPE", "IMAGE", "STRING")
    RETURN_NAMES = ("pipe", "image", "resolve_image_log")
    FUNCTION = "resolve_image"
    CATEGORY = "CMK/Developer/Legacy"

    def resolve_image(self, pipe, mode):
        if mode == "Original":
            image_out = first_not_none(pipe.get("image_original"), pipe.get("image_1st_pass"), pipe.get("image"))
            log = "Original | original image"
        else:
            image_out = first_not_none(pipe.get("image"), pipe.get("image_1st_pass"), pipe.get("image_original"))
            log = f"{mode} | current pipe image"

        new_pipe = dict(pipe)
        new_pipe["image"] = image_out
        return (new_pipe, image_out, log)
