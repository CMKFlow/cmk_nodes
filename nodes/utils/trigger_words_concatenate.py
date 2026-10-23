class CMKTriggerWordsConcatenate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger_words_1": ("STRING", {"forceInput": True}),
                "trigger_words_2": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("trigger_words",)
    FUNCTION = "concatenate"
    CATEGORY = "CMK/Toolbox/Text"

    def concatenate(self, **kwargs):
        first = kwargs.get("trigger_words_1") or ""
        second = kwargs.get("trigger_words_2") or ""
        if not first:
            return (second,)
        if not second:
            return (first,)
        return (f"{first},, {second}",)
