from __future__ import annotations

import os

import folder_paths
import comfy.sd
import comfy.utils

from ..pipe.cmk_log_pipe import cmk_add_block, cmk_bool


class CMKCheckpointVAELoader:
    """Load checkpoint model/clip and resolve the VAE in one compact CMK utility node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "vae_name": (folder_paths.get_filename_list("vae"),),
                "checkpoint_vae": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CMK_LOG_PIPE")
    RETURN_NAMES = ("model", "clip", "vae", "log_pipe")
    FUNCTION = "load_checkpoint_vae"
    CATEGORY = "CMK/Toolbox/Model & LoRA"

    def load_checkpoint_vae(self, ckpt_name, vae_name, checkpoint_vae):
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        checkpoint_data = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        model = checkpoint_data[0]
        clip = checkpoint_data[1]
        checkpoint_vae_obj = checkpoint_data[2]

        if bool(checkpoint_vae):
            vae = checkpoint_vae_obj
            vae_source = "Checkpoint"
            active_vae_name = "internal"
        else:
            vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
            vae_sd = comfy.utils.load_torch_file(vae_path)
            vae = comfy.sd.VAE(sd=vae_sd)
            vae_source = "External"
            active_vae_name = vae_name

        log_lines = [
            f"Checkpoint : {ckpt_name}",
            f"VAE Source : {vae_source}",
            f"VAE        : {active_vae_name}",
        ]
        log_pipe = cmk_add_block({"blocks": []}, "Checkpoint VAE Loader", 5, log_lines, True)
        return (model, clip, vae, log_pipe)


def load_checkpoint_vae_resources(ckpt_name, vae_name, checkpoint_vae):
    """Load shared model resources without coupling them to a PROCESS pipe."""
    ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
    checkpoint_data = comfy.sd.load_checkpoint_guess_config(
        ckpt_path,
        output_vae=True,
        output_clip=True,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )
    model = checkpoint_data[0]
    clip = checkpoint_data[1]
    checkpoint_vae_obj = checkpoint_data[2]

    if bool(checkpoint_vae):
        vae = checkpoint_vae_obj
        vae_source = "checkpoint"
        active_vae_name = "internal"
    else:
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        vae_sd = comfy.utils.load_torch_file(vae_path)
        vae = comfy.sd.VAE(sd=vae_sd)
        vae_source = "external"
        active_vae_name = str(vae_name)

    metadata = {
        "ckpt_name": str(ckpt_name),
        "vae_name": active_vae_name,
        "checkpoint_vae": bool(checkpoint_vae),
        "vae_source": vae_source,
        "model_pipe_source": "CMK Checkpoint VAE Loader -Pipe-",
    }
    return model, clip, vae, metadata
