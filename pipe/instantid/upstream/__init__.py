"""Vendored InstantID runtime used internally by CMK.

The implementation originates from cubiq/ComfyUI_InstantID and is distributed
under Apache-2.0. See LICENSE and NOTICE in this directory.
"""

from .InstantID import ApplyInstantIDAdvanced, InstantIDFaceAnalysis, InstantIDModelLoader

CMK_INSTANTID_CLASSES = {
    "CMKInternalInstantIDModelLoader": InstantIDModelLoader,
    "CMKInternalInstantIDFaceAnalysis": InstantIDFaceAnalysis,
    "CMKInternalApplyInstantIDAdvanced": ApplyInstantIDAdvanced,
}

__all__ = ["CMK_INSTANTID_CLASSES"]
