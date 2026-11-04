import os
from datetime import datetime

import folder_paths
from comfy_api.latest import io, ui, Input, Types


class CMK_SaveProjectVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CMK_SaveProjectVideo",
            display_name="CMK Save Project Video",
            category="CMK/Toolbox/I-O",
            inputs=[
                io.Video.Input("video"),
                io.Boolean.Input("save_enabled", default=True),
                io.String.Input("filename_prefix", default="video"),
                io.String.Input("output_folder", default=""),
                io.Boolean.Input("use_date_folder", default=True),
                io.String.Input("caption", default="", multiline=True),
                io.Combo.Input("format", options=Types.VideoContainer.as_input(), default="auto"),
                io.Combo.Input("codec", options=Types.VideoCodec.as_input(), default="auto"),
            ],
            outputs=[
                io.Video.Output("video"),
                io.String.Output("filename"),
                io.String.Output("foldername"),
                io.String.Output("full_path"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        video: Input.Video,
        save_enabled: bool,
        filename_prefix: str,
        output_folder: str,
        use_date_folder: bool,
        caption: str,
        format: str,
        codec: str,
    ) -> io.NodeOutput:
        if not save_enabled:
            return io.NodeOutput(video, "", "", "")

        base_output = folder_paths.get_output_directory()

        parts = []
        if output_folder.strip():
            parts.append(output_folder.strip())

        if use_date_folder:
            parts.append(datetime.now().strftime("%Y-%m-%d"))

        target_folder = os.path.join(base_output, *parts)
        os.makedirs(target_folder, exist_ok=True)

        extension = Types.VideoContainer.get_extension(format)

        counter = 1
        while True:
            filename = f"{filename_prefix}_{counter:05d}_.{extension}"
            full_path = os.path.join(target_folder, filename)
            if not os.path.exists(full_path):
                break
            counter += 1

        video.save_to(
            full_path,
            format=Types.VideoContainer(format),
            codec=codec,
            metadata=None,
        )

        if caption is not None and str(caption).strip():
            txt_path = os.path.splitext(full_path)[0] + ".txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(str(caption))

        return io.NodeOutput(
            video,
            filename,
            target_folder,
            full_path,
            ui=ui.PreviewVideo([ui.SavedResult(filename, os.path.relpath(target_folder, base_output), io.FolderType.output)]),
        )


NODE_CLASS_MAPPINGS = {
    "CMK_SaveProjectVideo": CMK_SaveProjectVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CMK_SaveProjectVideo": "CMK Save Project Video",
}
