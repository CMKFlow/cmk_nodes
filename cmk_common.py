import comfy.samplers


SAMPLERS = comfy.samplers.KSampler.SAMPLERS
SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS

RESOLVE_LATENT_SOURCES = [
    "Auto",
    "Original",
    "Encoded",
]

RESOLVE_IMAGE_SOURCES = [
    "Auto",
    "Current",
    "Original",
    "Processed",
]


def first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def optional_bool(value, default=False):
    if value is None:
        return default
    return bool(value)
