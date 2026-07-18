from .pipe.cmk_pipe_image import (
    CMKPipeCreateImage,
    CMKPipePeekPreprocessImage,
    CMKPipePeekControlNetSource,
)
from .pipe.cmk_pipe_sampler import (
    CMKPipeSetSampler,
    CMKPipePeekKSampler,
    CMKPipePeekKSamplerRefinerSource,
    CMKPipeSetKSampler,
    CMKKSamplerPipe,
)
from .pipe.cmk_sampler_prepare import CMKSamplerPrepareSDXLPipe
from .pipe.cmk_process_forward import CMKProcessForwardPipe
from .pipe.cmk_image_forward import CMKImageForward
from .pipe.cmk_model_forward import CMKModelForwardPipe
from .pipe.cmk_refiner_prepare import CMKRefinerPrepareSDXLPipe
from .pipe.cmk_refiner import CMKRefinerPipe
from .pipe.cmk_refiner_boundary_cache import CMKRefinerBoundaryCache
from .pipe.cmk_module_boundary_cache import (
    CMKDetailerBoundaryCache,
    CMKFaceBoundaryCache,
    CMKFaceSwapBoundaryCache,
)
from .pipe.cmk_detailer_prepare import CMKDetailerPreparePipe
from .pipe.cmk_faceprocess_prepare import CMKFaceProcessPreparePipe
from .pipe.cmk_faceprocess import CMKFaceProcessPipe
from .pipe.cmk_detailer_finalize import CMKDetailerFinalizePipe
from .pipe.cmk_pipe_process import (
    CMKPipeCreateDetailer,
    CMKPipeCreateFaceProcess,
    CMKPipePeekPreprocessDetailer,
    CMKPipePeekPreprocessFace,
    CMKPipeSetDetailer,
    CMKPipePeekDetailer,
    CMKPipePeekFaceProcess,
    CMKPipeSetDetailerResult,
    CMKPipeSetFaceResult,
    CMKPipeSetFaceProcessResult,
)
from .pipe.cmk_pipe_refiner import (
    CMKPipeSetRefiner,
    CMKPipePeekPreprocessRefiner,
    CMKPipeCreateRefiner,
    CMKPipePeekRefiner,
)
from .nodes.image.empty_image_mask import CMK_EmptyImageMask
from .nodes.image.image_mask_switch import CMK_ImageMaskSwitch
from .nodes.image.image_metrics import CMK_ImageMetrics, CMK_ImageQuickMetrics
from .nodes.image.smart_detailer import CMK_SmartDetailer, CMK_SmartDetailerPipe
from .nodes.image.segs_concate import CMK_SEGSConcate
from .nodes.image.smart_outpaint_pad import CMK_SmartOutpaintPad
from .nodes.image.smart_upscale import CMK_SmartUpscaler, CMK_SmartUpscalerPipe
from .nodes.swap.face_process import CMK_FaceProcess
from .nodes.io.filename_tools import CMK_FilenameBase
from .nodes.io.source_path_info import CMKSourcePathInfo
from .nodes.io.save_project_image import CMK_SaveProjectImage
from .nodes.io.save_project_text import CMK_SaveProjectText
from .nodes.io.save_project_video import CMK_SaveProjectVideo
from .nodes.video.video_metrics import CMK_VideoMetrics, CMK_VideoQuickMetrics
from .nodes.video.split_video_segments import CMKSplitVideoIntoSegments
from .nodes.video.face_swap_video_loader import CMKFaceSwapVideoLoader
from .nodes.video.merge_and_save_video import CMKMergeAndSaveVideo
from .nodes.video.video_compare import CMKVideoCompare
from .nodes.video.face_swap_video import CMKFaceSwapVideo
from .pipe.cmk_get_pipe import CMKGetPipe
from .pipe.cmk_pipe_debug import CMKPipeInspect
from .pipe.cmk_log_pipe import (
    CMKLogCreate,
    CMKLogSetBlock,
    CMKLogExportText,
    CMKLogConcat,
)
from .nodes.controlnet.controlnet import (
    CMKControlNetPrepare,
    CMKPipeSetControlNet,
)
from .pipe.controlnet.cmk_controlnet_prepare import CMKControlNetPreparePipe
from .loader.checkpoint_vae_loader import CMKCheckpointVAELoader
from .pipe.loaders.checkpoint_vae_loader import CMKCheckpointVAELoaderPipe
from .pipe.loaders.cmk_load_image import CMKLoadImage
from .pipe.loaders.cmk_image_load_resize import CMKImageLoadAndResizePipe
from .pipe.loaders.cmk_swap_image_loader import CMKSwapImageLoaderPipe
from .loader.cmk_lora_text_loader import CMKLoRATextLoader


# FaceSwap / Diagnostic infrastructure
from .nodes.utils.face_crop import CMKFaceCrop
from .nodes.swap.face_select import CMKFaceSelect
from .nodes.utils.preview_render import CMKPreviewRender
from .nodes.utils.preview_board import CMKPreviewBoard
from .nodes.utils.summary import CMKSummary
from .nodes.swap.face_mask import CMKFaceMask
from .nodes.swap.face_restore import CMKFaceRestore
from .nodes.swap.face_swap import CMKFaceSwapImage, CMKFaceSwapImagePipe


NODE_CLASS_MAPPINGS = {
    # Utils / Loaders
    "CMKCheckpointVAELoader": CMKCheckpointVAELoader,
    "CMKCheckpointVAELoaderPipe": CMKCheckpointVAELoaderPipe,
    "CMKLoadImage": CMKLoadImage,
    "CMKImageLoadAndResizePipe": CMKImageLoadAndResizePipe,
    "CMKSwapImageLoaderPipe": CMKSwapImageLoaderPipe,
    "CMKLoRATextLoader": CMKLoRATextLoader,

    # Pipe / Image
    "CMKGetPipe": CMKGetPipe,
    "CMKPipeInspect": CMKPipeInspect,
    "CMKPipeCreateImage": CMKPipeCreateImage,
    "CMKPipePeekPreprocessImage": CMKPipePeekPreprocessImage,
    "CMKPipePeekControlNetSource": CMKPipePeekControlNetSource,

    # Pipe / Sampler
    "CMKPipeSetSampler": CMKPipeSetSampler,
    "CMKSamplerPrepareSDXLPipe": CMKSamplerPrepareSDXLPipe,
    "CMKPipePeekKSampler": CMKPipePeekKSampler,
    "CMKPipePeekKSamplerRefinerSource": CMKPipePeekKSamplerRefinerSource,
    "CMKPipeSetKSampler": CMKPipeSetKSampler,
    "CMKKSamplerPipe": CMKKSamplerPipe,
    "CMKProcessForwardPipe": CMKProcessForwardPipe,
    "CMKImageForward": CMKImageForward,
    "CMKModelForwardPipe": CMKModelForwardPipe,

    # Pipe / Process
    "CMKPipeCreateDetailer": CMKPipeCreateDetailer,
    "CMKDetailerPreparePipe": CMKDetailerPreparePipe,
    "CMKFaceProcessPreparePipe": CMKFaceProcessPreparePipe,
    "CMKFaceProcessPipe": CMKFaceProcessPipe,
    "CMKDetailerFinalizePipe": CMKDetailerFinalizePipe,
    "CMKPipeCreateFaceProcess": CMKPipeCreateFaceProcess,
    "CMKPipePeekPreprocessDetailer": CMKPipePeekPreprocessDetailer,
    "CMKPipePeekPreprocessFace": CMKPipePeekPreprocessFace,
    "CMKPipeSetDetailer": CMKPipeSetDetailer,
    "CMKPipePeekDetailer": CMKPipePeekDetailer,
    "CMKPipePeekFaceProcess": CMKPipePeekFaceProcess,
    "CMKPipeSetDetailerResult": CMKPipeSetDetailerResult,
    "CMKPipeSetFaceResult": CMKPipeSetFaceResult,
    "CMKPipeSetFaceProcessResult": CMKPipeSetFaceProcessResult,

    # Pipe / Refiner
    "CMKPipeSetRefiner": CMKPipeSetRefiner,
    "CMKPipePeekPreprocessRefiner": CMKPipePeekPreprocessRefiner,
    "CMKPipeCreateRefiner": CMKPipeCreateRefiner,
    "CMKPipePeekRefiner": CMKPipePeekRefiner,
    "CMKRefinerPrepareSDXLPipe": CMKRefinerPrepareSDXLPipe,
    "CMKRefinerPipe": CMKRefinerPipe,
    "CMKRefinerBoundaryCache": CMKRefinerBoundaryCache,
    "CMKDetailerBoundaryCache": CMKDetailerBoundaryCache,
    "CMKFaceBoundaryCache": CMKFaceBoundaryCache,
    "CMKFaceSwapBoundaryCache": CMKFaceSwapBoundaryCache,


    # Standalone CMK Nodes
    "CMK_EmptyImageMask": CMK_EmptyImageMask,
    "CMK_ImageMaskSwitch": CMK_ImageMaskSwitch,
    "CMK_ImageMetrics": CMK_ImageMetrics,
    "CMK_ImageQuickMetrics": CMK_ImageQuickMetrics,
    "CMK_SmartDetailer": CMK_SmartDetailer,
    "CMK_SmartDetailerPipe": CMK_SmartDetailerPipe,
    "CMK_SEGSConcate": CMK_SEGSConcate,
    "CMK_SmartOutpaintPad": CMK_SmartOutpaintPad,
    "CMK_SmartUpscaler": CMK_SmartUpscaler,
    "CMK_SmartUpscalerPipe": CMK_SmartUpscalerPipe,
    "CMK_FaceProcess": CMK_FaceProcess,
    "CMK_FilenameBase": CMK_FilenameBase,
    "CMK_SourcePathInfo": CMKSourcePathInfo,
    "CMK_SaveProjectImage": CMK_SaveProjectImage,
    "CMK_SaveProjectText": CMK_SaveProjectText,
    "CMK_SaveProjectVideo": CMK_SaveProjectVideo,
    "CMK_VideoMetrics": CMK_VideoMetrics,
    "CMK_VideoQuickMetrics": CMK_VideoQuickMetrics,
    "CMKSplitVideoIntoSegments": CMKSplitVideoIntoSegments,
    "CMKFaceSwapVideoLoader": CMKFaceSwapVideoLoader,
    "CMKMergeAndSaveVideo": CMKMergeAndSaveVideo,
    "CMKVideoCompare": CMKVideoCompare,
    "CMKFaceSwapVideo": CMKFaceSwapVideo,

    # ControlNet
    "CMKControlNetPrepare": CMKControlNetPrepare,
    "CMKControlNetPreparePipe": CMKControlNetPreparePipe,

    # Pipe / ControlNet
    "CMKPipeSetControlNet": CMKPipeSetControlNet,

    # Smart Pipe

    # Log Pipe
    "CMKLogCreate": CMKLogCreate,
    "CMKLogSetBlock": CMKLogSetBlock,
    "CMKLogExportText": CMKLogExportText,
    "CMKLogConcat": CMKLogConcat,

    # FaceSwap / Diagnostic
    "CMKFaceSwapImage": CMKFaceSwapImage,
    "CMKFaceSwapImagePipe": CMKFaceSwapImagePipe,
        "CMKFaceCrop": CMKFaceCrop,
        "CMKFaceSelect": CMKFaceSelect,
        "CMKPreviewRender": CMKPreviewRender,
        "CMKPreviewBoard": CMKPreviewBoard,
        "CMKSummary": CMKSummary,
        "CMKFaceMask": CMKFaceMask,
        "CMKFaceRestore": CMKFaceRestore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CMKCheckpointVAELoader": "CMK Checkpoint VAE Loader",
    "CMKCheckpointVAELoaderPipe": "CMK Flow · Checkpoint & VAE",
    "CMKLoadImage": "CMK Flow · Load Image",
    "CMKImageLoadAndResizePipe": "CMK Flow · Image Input",
    "CMKSwapImageLoaderPipe": "CMK Flow · FaceSwap Image Input",
    "CMKLoRATextLoader": "CMK LoRA Text Loader",

    "CMKGetPipe": "CMK Pipe Get",
    "CMKPipeInspect": "CMK Pipe Inspect",
    "CMKPipeCreateImage": "CMK Flow · 01 START HERE · Create Image",
    "CMKPipePeekPreprocessImage": "CMK Pipe Peek Preprocess Image",
    "CMKPipePeekControlNetSource": "CMK Pipe Peek ControlNet Source",

    "CMKPipeSetSampler": "CMK Pipe Set Sampler",
    "CMKSamplerPrepareSDXLPipe": "CMK Sampler Prepare SDXL -Pipe-",
    "CMKPipePeekKSampler": "CMK Pipe Peek KSampler",
    "CMKPipePeekKSamplerRefinerSource": "CMK Pipe Peek KSampler Refiner Source",
    "CMKPipeSetKSampler": "CMK Pipe Set KSampler",
    "CMKKSamplerPipe": "CMK KSampler -Pipe-",
    "CMKProcessForwardPipe": "CMK Process Forward -Pipe-",
    "CMKImageForward": "CMK Image Forward -Pipe-",
    "CMKModelForwardPipe": "CMK Model Forward -Pipe-",

    "CMKPipeCreateDetailer": "CMK Pipe Create Detailer",
    "CMKDetailerPreparePipe": "CMK Detailer Prepare -Pipe-",
    "CMKFaceProcessPreparePipe": "CMK FaceProcess Prepare -Pipe-",
    "CMKFaceProcessPipe": "CMK FaceProcess -Pipe-",
    "CMKDetailerFinalizePipe": "CMK Detailer Finalize -Pipe-",
    "CMKPipeCreateFaceProcess": "CMK Pipe Create Face Process",
    "CMKPipePeekPreprocessDetailer": "CMK Pipe Peek Preprocess Detailer",
    "CMKPipePeekPreprocessFace": "CMK Pipe Peek Preprocess Face",
    "CMKPipeSetDetailer": "CMK Pipe Set Detailer",
    "CMKPipePeekDetailer": "CMK Pipe Peek Detailer",
    "CMKPipePeekFaceProcess": "CMK Pipe Peek Face Process",
    "CMKPipeSetDetailerResult": "CMK Pipe Set Detailer Result",
    "CMKPipeSetFaceResult": "CMK Pipe Set Face Result",
    "CMKPipeSetFaceProcessResult": "CMK Pipe Set Face Process Result",

    "CMKPipeSetRefiner": "CMK Pipe Set Refiner",
    "CMKPipePeekPreprocessRefiner": "CMK Pipe Peek Preprocess Refiner",
    "CMKPipeCreateRefiner": "CMK Pipe Create Refiner",
    "CMKPipePeekRefiner": "CMK Pipe Peek Refiner",
    "CMKRefinerPrepareSDXLPipe": "CMK Refiner Prepare SDXL -Pipe-",
    "CMKRefinerPipe": "CMK Refiner -Pipe-",
    "CMKRefinerBoundaryCache": "CMK Boundary Cache",
    "CMKDetailerBoundaryCache": "CMK Boundary Cache",
    "CMKFaceBoundaryCache": "CMK Boundary Cache",
    "CMKFaceSwapBoundaryCache": "CMK Boundary Cache",


    "CMK_EmptyImageMask": "CMK Empty Image Mask",
    "CMK_ImageMaskSwitch": "CMK Image and Mask Switch",
    "CMK_ImageMetrics": "CMK Image Metrics",
    "CMK_ImageQuickMetrics": "CMK Image QuickMetrics",
    "CMK_SmartDetailer": "CMK Smart Detailer",
    "CMK_SmartDetailerPipe": "CMK Smart Detailer -Pipe-",
    "CMK_SEGSConcate": "CMK SEGS CONCAT",
    "CMK_SmartOutpaintPad": "CMK Smart Outpaint Pad",
    "CMK_SmartUpscaler": "CMK Smart Upscaler",
    "CMK_SmartUpscalerPipe": "CMK Smart Upscaler -Pipe-",
    "CMK_FaceProcess": "CMK FaceProcess",
    "CMK_FilenameBase": "CMK Filename Base",
    "CMK_SourcePathInfo": "CMK SourcePathInfo",
    "CMK_SaveProjectImage": "CMK Flow · Save Project Image",
    "CMK_SaveProjectText": "CMK Save Project Text",
    "CMK_SaveProjectVideo": "CMK Save Project Video",
    "CMK_VideoMetrics": "CMK Video Metrics",
    "CMK_VideoQuickMetrics": "CMK Video QuickMetrics",
    "CMKSplitVideoIntoSegments": "CMK Split Video into Segments",
    "CMKFaceSwapVideoLoader": "CMK FaceSwap Video Loader",
    "CMKMergeAndSaveVideo": "CMK Merge and Save Video",
    "CMKVideoCompare": "CMK Video Compare",
    "CMKFaceSwapVideo": "CMK FaceSwap Video",

    "CMKControlNetPrepare": "CMK ControlNet Prepare",
    "CMKControlNetPreparePipe": "CMK Flow · 05 ControlNet (optional)",
    "CMKPipeSetControlNet": "CMK Pipe Set ControlNet",

    "CMKLogCreate": "CMK Log Create",
    "CMKLogSetBlock": "CMK Log Set Block",
    "CMKLogExportText": "CMK Log Export Text",
    "CMKLogConcat": "CMK LOG CONCAT",


    # FaceSwap / Diagnostic
    "CMKFaceSwapImage": "CMK FaceSwap Image",
    "CMKFaceSwapImagePipe": "CMK FaceSwap Image -Pipe-",
        "CMKFaceCrop": "CMK Face Crop",
        "CMKFaceSelect": "CMK Face Select",
        "CMKPreviewRender": "CMK Preview Render",
        "CMKPreviewBoard": "CMK Preview Board",
        "CMKSummary": "CMK Summary",
        "CMKFaceMask": "CMK Face Mask",
        "CMKFaceRestore": "CMK Face Restore",
}
