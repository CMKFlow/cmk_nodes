class CMKImageForward:
    """Read-only IMAGE throughpass for CMK frontend subgraphs.

    This node exists solely to expose the incoming IMAGE at a subgraph output
    without routing it through a processing Prepare or Execute node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"IMAGE": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "forward"
    CATEGORY = "CMK/Developer/Pipe/Forward"

    @staticmethod
    def forward(IMAGE):
        if IMAGE is None:
            raise ValueError("CMK Image Forward -Pipe-: IMAGE is missing")
        return (IMAGE,)
