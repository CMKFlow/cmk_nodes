from __future__ import annotations

import folder_paths

from ...loader.checkpoint_vae_loader import load_checkpoint_vae_resources


DEFAULT_CHECKPOINT = "juggernautXL_ragnarok.safetensors"


def _checkpoint_choices():
    choices = list(folder_paths.get_filename_list("checkpoints"))
    if DEFAULT_CHECKPOINT in choices:
        choices.remove(DEFAULT_CHECKPOINT)
        choices.insert(0, DEFAULT_CHECKPOINT)
    return choices


class CMKCheckpointVAELoaderPipe:
    """Create the shared read-only CMK MODEL resource pipe.

    Public contract:
        no PROCESS input
        MODEL output only

    Downstream modules may read model/clip/vae, but must never mutate MODEL
    or write module working state back into it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (_checkpoint_choices(),),
                "vae_name": (folder_paths.get_filename_list("vae"),),
                "checkpoint_vae": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CMK_MODEL_PIPE",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "load_checkpoint_vae_pipe"
    CATEGORY = "CMK/Flow/Input"

    def load_checkpoint_vae_pipe(self, ckpt_name, vae_name, checkpoint_vae):
        model, clip, vae, metadata = load_checkpoint_vae_resources(
            ckpt_name, vae_name, checkpoint_vae
        )
        return ({
            "model": model,
            "clip": clip,
            "vae": vae,
            **metadata,
        },)
