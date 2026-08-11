class CMKProcessForwardPipe:
    """Read-only PROCESS throughpass for CMK frontend subgraphs.

    This node exists solely to expose the incoming PROCESS pipe at a subgraph
    output without routing it through a processing Execute node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"PROCESS": ("CMK_PROCESS_SDXL",)}}

    RETURN_TYPES = ("CMK_PROCESS_SDXL",)
    RETURN_NAMES = ("PROCESS",)
    FUNCTION = "forward"
    CATEGORY = "CMK/Developer/Pipe/Forward"

    @staticmethod
    def forward(PROCESS):
        if PROCESS is None:
            return (None,)
        if not isinstance(PROCESS, dict):
            raise TypeError(
                "CMK Process Forward -Pipe-: PROCESS must be a CMK process pipe "
                f"(dict), got {type(PROCESS).__name__}"
            )
        return (PROCESS,)
