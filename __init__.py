from .nodes import WanMoeKSampler,WanMoeKSamplerAdvanced,SplitSigmasAtT

NODE_CLASS_MAPPINGS = {
    "WanMoeKSampler3pass":WanMoeKSampler,
    "WanMoeKSamplerAdvanced3pass":WanMoeKSamplerAdvanced,
    "SplitSigmasAtT":SplitSigmasAtT
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanMoeKSampler3pass": "Wan 3pass MoE KSampler",
    "WanMoeKSamplerAdvanced3pass": "Wan 3pass MoE KSampler (Advanced)",
    "SplitSigmasAtT": "Split sigmas at timestep"
}