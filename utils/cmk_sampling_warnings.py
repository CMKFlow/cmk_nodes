from __future__ import annotations

import warnings
from contextlib import contextmanager


_TORCHSDE_BOUNDARY_ROUNDING = (
    r"Should have (?:ta>=t0|tb<=t1) but got "
    r"(?:ta|tb)=[0-9.eE+-]+ and t[01]=[0-9.eE+-]+\."
)


@contextmanager
def ignore_torchsde_boundary_rounding():
    """Hide only torchsde's harmless float-boundary clamp warnings.

    BrownianInterval emits these messages immediately before clamping the
    queried Python float to the interval boundary.  Filtering the exact
    warning leaves the Brownian query and its result unchanged.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=_TORCHSDE_BOUNDARY_ROUNDING,
            category=UserWarning,
            module=r"torchsde\._brownian\.brownian_interval",
        )
        yield
