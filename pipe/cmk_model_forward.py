class CMKModelForwardPipe:
    """Read-only MODEL throughpass for CMK frontend subgraphs.

    This node exists solely to expose the incoming MODEL resource pipe at a
    subgraph output without routing it through a Prepare or Execute node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"MODEL": ("CMK_MODEL_PIPE",)}}

    RETURN_TYPES = ("CMK_MODEL_PIPE",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "forward"
    CATEGORY = "CMK/Developer/Pipe/Forward"

    @staticmethod
    def forward(MODEL):
        if MODEL is None:
            raise ValueError("CMK Model Forward -Pipe-: MODEL is missing")
        if not isinstance(MODEL, dict):
            raise TypeError(
                "CMK Model Forward -Pipe-: MODEL must be a CMK model pipe "
                f"(dict), got {type(MODEL).__name__}"
            )
        return (MODEL,)
