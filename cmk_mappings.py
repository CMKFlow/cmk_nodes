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
from .pipe.cmk_regional_conditioning import CMKRegionalConditioningSDXL
from .pipe.cmk_process_forward import CMKProcessForwardPipe
from .pipe.cmk_visual import CMKVisualPass, CMKVisualProvider, CMKVisualizer
from .pipe.cmk_family_result import (
    CMKImageCompareEnableGate,
    CMKModuleBypassGate,
    CMKControlNetBypassGate,
    CMKZITControlNetBypassGate,
    CMKCombinedControlNetBypassGate,
    CMKSamplerBypassGate,
    CMKProcessEnableFlag,
    CMKFamilyBranchGateSDXL,
    CMKFamilyBranchGateSDXLSampled,
    CMKFamilyBranchGateZImage,
    CMKSDXLResultBridgePipe,
    CMKFamilyResultMergePipe,
    CMKResultToLegacyBridgePipe,
    CMKZImageProcessForwardPipe,
    CMKResultProcessForwardPipe,
    CMKResultUnpackPipe,
    CMKResultPackPipe,
)
from .pipe.cmk_image_forward import CMKImageForward, CMKImagePreviewForward
from .pipe.cmk_model_forward import CMKModelForwardPipe
from .pipe.cmk_refiner_prepare import CMKRefinerPrepareSDXLPipe
from .pipe.cmk_refiner import CMKRefinerPipe
from .pipe.cmk_refiner_boundary_cache import CMKRefinerBoundaryCache
from .pipe.cmk_module_boundary_cache import (
    CMKDetailerBoundaryCache,
    CMKFaceBoundaryCache,
    CMKFaceRebuildBoundaryCache,
    CMKFaceSwapBoundaryCache,
    CMKUpscaleSaveBoundaryCache,
    CMKZImageBoundaryCache,
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
from .pipe.controlnet.cmk_zit_controlnet_prepare import CMKZITControlNetPreparePipe
from .pipe.controlnet.cmk_combined_controlnet_prepare import (
    CMKCombinedControlNetPreparePipe,
)
from .pipe.instantid.cmk_instantid_sampler import (
    CMKInstantIDBoundary,
    CMKInstantIDSamplerSDXLPipe,
)
from .nodes.instantid_face_rebuild import (
    CMKInstantIDFaceRebuildPasteback,
    CMKInstantIDFaceRebuildPastebackPipe,
    CMKInstantIDFaceRebuildPrepare,
    CMKInstantIDFaceRebuildPreparePipe,
)
from .nodes.instantid_face_detailer import CMKInstantIDFaceDetailerSDXL
from .nodes.instantid_face_rebuild_advanced import (
    CMKInstantIDFaceRebuildAdvancedSDXL,
    CMKInstantIDFaceRebuildSDXL,
)
from .loader.checkpoint_vae_loader import CMKCheckpointVAELoader
from .pipe.loaders.checkpoint_vae_loader import CMKCheckpointVAELoaderPipe
from .pipe.loaders.z_image_turbo_loader import CMKZImageTurboLoaderPipe
from .pipe.cmk_z_image_turbo import (
    CMKSamplerPrepareZImageTurboPipe,
    CMKZImageTurboFinalizePipe,
)
from .pipe.loaders.cmk_load_image import CMKLoadImage
from .pipe.loaders.cmk_image_load_resize import CMKImageLoadAndResizePipe
from .pipe.loaders.cmk_swap_image_loader import CMKSwapImageLoaderPipe
from .loader.cmk_lora_text_loader import CMKLoRATextLoader


# FaceSwap / Diagnostic infrastructure
from .nodes.utils.face_crop import CMKFaceCrop
from .nodes.swap.face_select import CMKFaceSelect
from .nodes.utils.preview_render import CMKPreviewRender
from .nodes.utils.preview_board import CMKPreviewBoard
from .nodes.utils.diagnostic_concat import CMKDiagnosticConcat
from .nodes.utils.summary import CMKSummary
from .nodes.utils.native_flow_helpers import (
    CMKImageCompare,
    CMKSEGSPreview,
    CMKLoRAStackBuilder,
    CMKPromptConcat,
    CMKTriggerWordsFilter,
    CMKStringDual,
)
from .nodes.swap.face_mask import CMKFaceMask
from .nodes.swap.face_restore import CMKFaceRestore
from .nodes.swap.face_swap import CMKFaceSwapImage, CMKFaceSwapImagePipe


NODE_CLASS_MAPPINGS = {
    "CMKVisualPass": CMKVisualPass,
    "CMKVisualProvider": CMKVisualProvider,
    "CMKVisualizer": CMKVisualizer,
    "CMKImageCompareEnableGate": CMKImageCompareEnableGate,
    "CMKModuleBypassGate": CMKModuleBypassGate,
    "CMKControlNetBypassGate": CMKControlNetBypassGate,
    "CMKZITControlNetBypassGate": CMKZITControlNetBypassGate,
    "CMKCombinedControlNetBypassGate": CMKCombinedControlNetBypassGate,
    "CMKSamplerBypassGate": CMKSamplerBypassGate,
    "CMKProcessEnableFlag": CMKProcessEnableFlag,
    "CMKFamilyBranchGateSDXL": CMKFamilyBranchGateSDXL,
    "CMKFamilyBranchGateSDXLSampled": CMKFamilyBranchGateSDXLSampled,
    "CMKFamilyBranchGateZImage": CMKFamilyBranchGateZImage,
    # Utils / Loaders
    "CMKCheckpointVAELoader": CMKCheckpointVAELoader,
    "CMKCheckpointVAELoaderPipe": CMKCheckpointVAELoaderPipe,
    "CMKZImageTurboLoaderPipe": CMKZImageTurboLoaderPipe,
    "CMKLoadImage": CMKLoadImage,
    "CMKImageLoadAndResizePipe": CMKImageLoadAndResizePipe,
    "CMKSwapImageLoaderPipe": CMKSwapImageLoaderPipe,
    "CMKLoRATextLoader": CMKLoRATextLoader,

    # Pipe / Image
    "CMKGetPipe": CMKGetPipe,
    "CMKPipeInspect": CMKPipeInspect,
    "CMKPipeCreateImage": CMKPipeCreateImage,
    "CMKRegionalConditioningSDXL": CMKRegionalConditioningSDXL,
    "CMKPipePeekPreprocessImage": CMKPipePeekPreprocessImage,
    "CMKPipePeekControlNetSource": CMKPipePeekControlNetSource,

    # Pipe / Sampler
    "CMKPipeSetSampler": CMKPipeSetSampler,
    "CMKSamplerPrepareSDXLPipe": CMKSamplerPrepareSDXLPipe,
    "CMKSamplerPrepareZImageTurboPipe": CMKSamplerPrepareZImageTurboPipe,
    "CMKZImageTurboFinalizePipe": CMKZImageTurboFinalizePipe,
    "CMKPipePeekKSampler": CMKPipePeekKSampler,
    "CMKPipePeekKSamplerRefinerSource": CMKPipePeekKSamplerRefinerSource,
    "CMKPipeSetKSampler": CMKPipeSetKSampler,
    "CMKKSamplerPipe": CMKKSamplerPipe,
    "CMKProcessForwardPipe": CMKProcessForwardPipe,
    "CMKSDXLResultBridgePipe": CMKSDXLResultBridgePipe,
    "CMKFamilyResultMergePipe": CMKFamilyResultMergePipe,
    "CMKResultToLegacyBridgePipe": CMKResultToLegacyBridgePipe,
    "CMKZImageProcessForwardPipe": CMKZImageProcessForwardPipe,
    "CMKResultProcessForwardPipe": CMKResultProcessForwardPipe,
    "CMKResultUnpackPipe": CMKResultUnpackPipe,
    "CMKResultPackPipe": CMKResultPackPipe,
    "CMKImageForward": CMKImageForward,
    "CMKImagePreviewForward": CMKImagePreviewForward,
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
    "CMKFaceRebuildBoundaryCache": CMKFaceRebuildBoundaryCache,
    "CMKFaceSwapBoundaryCache": CMKFaceSwapBoundaryCache,
    "CMKUpscaleSaveBoundaryCache": CMKUpscaleSaveBoundaryCache,
    "CMKZImageBoundaryCache": CMKZImageBoundaryCache,


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
    "CMKZITControlNetPreparePipe": CMKZITControlNetPreparePipe,
    "CMKCombinedControlNetPreparePipe": CMKCombinedControlNetPreparePipe,
    "CMKInstantIDSamplerSDXLPipe": CMKInstantIDSamplerSDXLPipe,
    "CMKInstantIDBoundary": CMKInstantIDBoundary,
    "CMKInstantIDFaceRebuildPrepare": CMKInstantIDFaceRebuildPrepare,
    "CMKInstantIDFaceRebuildPasteback": CMKInstantIDFaceRebuildPasteback,
    "CMKInstantIDFaceRebuildPreparePipe": CMKInstantIDFaceRebuildPreparePipe,
    "CMKInstantIDFaceRebuildPastebackPipe": CMKInstantIDFaceRebuildPastebackPipe,
    "CMKInstantIDFaceDetailerSDXL": CMKInstantIDFaceDetailerSDXL,
    "CMKInstantIDFaceRebuildSDXL": CMKInstantIDFaceRebuildSDXL,
    "CMKInstantIDFaceRebuildAdvancedSDXL": CMKInstantIDFaceRebuildAdvancedSDXL,

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
        "CMKDiagnosticConcat": CMKDiagnosticConcat,
        "CMKSummary": CMKSummary,
        "CMKImageCompare": CMKImageCompare,
        "CMKSEGSPreview": CMKSEGSPreview,
        "CMKLoRAStackBuilder": CMKLoRAStackBuilder,
        "CMKPromptConcat": CMKPromptConcat,
        "CMKTriggerWordsFilter": CMKTriggerWordsFilter,
        "CMKStringDual": CMKStringDual,
        "CMKFaceMask": CMKFaceMask,
        "CMKFaceRestore": CMKFaceRestore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CMKVisualPass": "CMK Visual Forward -Pipe-",
    "CMKVisualProvider": "CMK Visual Provider -Pipe-",
    "CMKVisualizer": "CMK Flow · 100 Visualizer",
    "CMKCheckpointVAELoader": "CMK Checkpoint VAE Loader",
    "CMKCheckpointVAELoaderPipe": "CMK Flow · Checkpoint & VAE",
    "CMKZImageTurboLoaderPipe": "CMK Z-Image Turbo Loader -Pipe-",
    "CMKLoadImage": "CMK Flow · Load Image",
    "CMKImageLoadAndResizePipe": "CMK Flow · Image Input",
    "CMKSwapImageLoaderPipe": "CMK FaceSwap Image Input",
    "CMKLoRATextLoader": "CMK LoRA Text Loader",
    "CMKImageCompare": "CMK Image Compare",
    "CMKSEGSPreview": "CMK SEGS Preview",
    "CMKLoRAStackBuilder": "CMK LoRA Stack Builder",
    "CMKPromptConcat": "CMK Prompt Concatenate",
    "CMKTriggerWordsFilter": "CMK Trigger Words Filter",
    "CMKStringDual": "CMK Positive / Negative Text",

    "CMKGetPipe": "CMK Pipe Get",
    "CMKPipeInspect": "CMK Pipe Inspect",
    "CMKPipeCreateImage": "CMK Flow · 01 START HERE · Create Image",
    "CMKRegionalConditioningSDXL": "CMK Flow · 02 Regional Conditioning SDXL",
    "CMKPipePeekPreprocessImage": "CMK Pipe Peek Preprocess Image",
    "CMKPipePeekControlNetSource": "CMK Pipe Peek ControlNet Source",

    "CMKPipeSetSampler": "CMK Pipe Set Sampler",
    "CMKSamplerPrepareSDXLPipe": "CMK Sampler Prepare SDXL -Pipe-",
    "CMKSamplerPrepareZImageTurboPipe": "CMK Sampler Prepare Z-Image Turbo -Pipe-",
    "CMKZImageTurboFinalizePipe": "CMK Z-Image Turbo Finalize -Pipe-",
    "CMKPipePeekKSampler": "CMK Pipe Peek KSampler",
    "CMKPipePeekKSamplerRefinerSource": "CMK Pipe Peek KSampler Refiner Source",
    "CMKPipeSetKSampler": "CMK Pipe Set KSampler",
    "CMKKSamplerPipe": "CMK KSampler -Pipe-",
    "CMKProcessForwardPipe": "CMK Process Forward -Pipe-",
    "CMKModuleBypassGate": "CMK Module Bypass Gate",
    "CMKControlNetBypassGate": "CMK ControlNet Bypass Gate",
    "CMKZITControlNetBypassGate": "CMK ZIT ControlNet Bypass Gate",
    "CMKCombinedControlNetBypassGate": "CMK Combined ControlNet Bypass Gate",
    "CMKSamplerBypassGate": "CMK Sampler Bypass Gate",
    "CMKProcessEnableFlag": "CMK Process Enable Flag",
    "CMKSDXLResultBridgePipe": "CMK SDXL Result Bridge -Pipe-",
    "CMKFamilyResultMergePipe": "CMK Flow · 35 Active Family Result",
    "CMKResultToLegacyBridgePipe": "CMK Result Bridge -Pipe-",
    "CMKZImageProcessForwardPipe": "CMK Z-Image Process Forward -Pipe-",
    "CMKResultProcessForwardPipe": "CMK Result Process Forward -Pipe-",
    "CMKResultUnpackPipe": "CMK Result Unpack -Pipe-",
    "CMKResultPackPipe": "CMK Result Pack -Pipe-",
    "CMKImageForward": "CMK Image Forward -Pipe-",
    "CMKImagePreviewForward": "CMK Image Preview Forward -Pipe-",
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
    "CMKFaceRebuildBoundaryCache": "CMK Boundary Cache",
    "CMKFaceSwapBoundaryCache": "CMK Boundary Cache",
    "CMKUpscaleSaveBoundaryCache": "CMK Boundary Cache",
    "CMKZImageBoundaryCache": "CMK Boundary Cache",


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
    "CMK_SaveProjectImage": "CMK Save Project Image",
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
    "CMKControlNetPreparePipe": "CMK Flow · 05 ControlNet SDXL",
    "CMKZITControlNetPreparePipe": "CMK Flow · 05 ControlNet ZIT",
    "CMKCombinedControlNetPreparePipe": "CMK Flow · 05 ControlNet Combined",
    "CMKInstantIDSamplerSDXLPipe": "CMK InstantID Sampler SDXL -Pipe-",
    "CMKInstantIDBoundary": "CMK InstantID Boundary",
    "CMKInstantIDFaceRebuildPrepare": "CMK InstantID Face Rebuild · Prepare",
    "CMKInstantIDFaceRebuildPasteback": "CMK InstantID Face Rebuild · Pasteback",
    "CMKInstantIDFaceRebuildPreparePipe": "CMK Flow · InstantID Face Rebuild · Prepare",
    "CMKInstantIDFaceRebuildPastebackPipe": "CMK Flow · InstantID Face Rebuild · Pasteback",
    "CMKInstantIDFaceDetailerSDXL": "CMK InstantID Face Detailer SDXL",
    "CMKInstantIDFaceRebuildSDXL": "CMK FaceRebuild SDXL",
    "CMKInstantIDFaceRebuildAdvancedSDXL": "CMK FaceRebuild SDXL · Advanced",
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
        "CMKDiagnosticConcat": "CMK Diagnostic Concat",
        "CMKSummary": "CMK Summary",
        "CMKFaceMask": "CMK Face Mask",
        "CMKFaceRestore": "CMK Face Restore",
}
