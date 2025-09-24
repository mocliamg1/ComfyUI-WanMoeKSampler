from .nodes import WanMoeKSampler,WanMoeKSamplerAdvanced

NODE_CLASS_MAPPINGS = {
    "WanMoeKSampler3pass":WanMoeKSampler,
    "WanMoeKSamplerAdvanced3pass":WanMoeKSamplerAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanMoeKSampler3pass": "Wan 3pass MoE KSampler",
    "WanMoeKSamplerAdvanced3pass": "Wan 3pass MoE KSampler (Advanced)"
}
