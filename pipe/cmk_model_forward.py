class CMKModelForwardPipe:
    """Read-only MODEL throughpass for CMK frontend subgraphs.

    This node exists solely to expose the incoming MODEL resource pipe at a
    subgraph output without routing it through a Prepare or Execute node.
    PostProcess modules never change MODEL (OD-2), so their public MODEL
    output is the forwarded input: requesting it evaluates no module work.
    MODEL is optional; without a checkpoint it stays ``None``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {"MODEL": ("CMK_MODEL_PIPE",)}}

    RETURN_TYPES = ("CMK_MODEL_PIPE",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "forward"
    CATEGORY = "CMK/Developer/Legacy Pipe"

    @staticmethod
    def forward(MODEL=None):
        if MODEL is not None and not isinstance(MODEL, dict):
            raise TypeError(
                "CMK Model Forward -Pipe-: MODEL must be a CMK model pipe "
                f"(dict), got {type(MODEL).__name__}"
            )
        return (MODEL,)
