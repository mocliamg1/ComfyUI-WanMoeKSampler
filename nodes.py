import torch

import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.model_sampling

import latent_preview


def wan_ksampler(model_high_noise, model_high_noise_2, model_low_noise, seed, steps, cfgs, sampler_name, scheduler, positive, negative, latent, boundary = 0.875, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, enable_second_high_pass=True):
    # boundary is .9 for i2v, .875 for t2v
    latent_image = latent["samples"]

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    assert start_step is None or start_step < steps
    assert last_step is None or last_step >= start_step
    if start_step is None:
        start_step = 0
    if last_step is None:
        last_step=9999

    # first, we get all sigmas
    sampling = model_high_noise.get_model_object("model_sampling")
    sigmas = comfy.samplers.calculate_sigmas(sampling,scheduler,steps)
    # why are timesteps 0-1000?
    timesteps = [sampling.timestep(sigma)/1000 for sigma in sigmas.tolist()]
    switching_step = steps
    for (i,t) in enumerate(timesteps[1:]):
        if t < boundary:
            switching_step = i
            break
    print(f"switching model at step {switching_step}")
    start_with_high = start_step<switching_step
    end_wth_low = last_step>=switching_step

    # HIGH region: from start_step up to switching_step
    if start_with_high:
        print("Running high noise models...")
        callback1 = latent_preview.prepare_callback(model_high_noise, steps)
        callback2 = latent_preview.prepare_callback(model_high_noise_2, steps)

        high_start = start_step
        high_end = min(last_step, switching_step)
        high_total = max(0, high_end - high_start)

        if high_total > 0:
            # if second high pass disabled, run the first high model for the whole high region
            if not enable_second_high_pass:
                latent_image = comfy.sample.fix_empty_latent_channels(model_high_noise, latent_image)
                latent_image = comfy.sample.sample(model_high_noise, noise, steps, cfgs[0], sampler_name, scheduler, positive, negative, latent_image,
                                            denoise=denoise, disable_noise=end_wth_low or disable_noise, start_step=high_start, last_step=high_end,
                                            force_full_denoise=end_wth_low or force_full_denoise, noise_mask=noise_mask, callback=callback1, disable_pbar=disable_pbar, seed=seed)
            else:
                # split high region into two halves between high model 1 and high model 2
                half = high_total // 2
                first_high_end = high_start + half
                second_high_begin = first_high_end

                # first high pass
                if first_high_end > high_start:
                    latent_image = comfy.sample.fix_empty_latent_channels(model_high_noise, latent_image)
                    latent_image = comfy.sample.sample(model_high_noise, noise, steps, cfgs[0], sampler_name, scheduler, positive, negative, latent_image,
                                                denoise=denoise, disable_noise=False, start_step=high_start, last_step=first_high_end,
                                                force_full_denoise=False, noise_mask=noise_mask, callback=callback1, disable_pbar=disable_pbar, seed=seed)

                # second high pass
                if high_end > second_high_begin:
                    latent_image = comfy.sample.fix_empty_latent_channels(model_high_noise_2, latent_image)
                    # this is the last high pass, so it should respect whether low region follows
                    latent_image = comfy.sample.sample(model_high_noise_2, noise, steps, cfgs[1], sampler_name, scheduler, positive, negative, latent_image,
                                                denoise=denoise, disable_noise=end_wth_low or disable_noise, start_step=second_high_begin, last_step=high_end,
                                                force_full_denoise=end_wth_low or force_full_denoise, noise_mask=noise_mask, callback=callback2, disable_pbar=disable_pbar, seed=seed)

    # LOW region: single low model covers from switching_step to end
    if end_wth_low:
        print("Running low noise model...")
        callback_low = latent_preview.prepare_callback(model_low_noise, steps)

        low_begin = max(start_step, switching_step)
        low_last = min(last_step, steps)
        low_total = max(0, low_last - low_begin)

        if low_total > 0:
            latent_image = comfy.sample.fix_empty_latent_channels(model_low_noise, latent_image)
            latent_image = comfy.sample.sample(model_low_noise, noise, steps, cfgs[2], sampler_name, scheduler, positive, negative, latent_image,
                                        denoise=denoise, disable_noise=disable_noise, start_step=low_begin, last_step=low_last,
                                        force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback_low, disable_pbar=disable_pbar, seed=seed)

    out = latent.copy()
    out["samples"] = latent_image
    return (out, )


def set_shift(model,sigma_shift):
    model_sampling = model.get_model_object("model_sampling")
    if not model_sampling:
        sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
        sampling_type = comfy.model_sampling.CONST
        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
    model_sampling.set_parameters(shift=sigma_shift, multiplier=1000)
    model.add_object_patch("model_sampling", model_sampling)
    return model

class WanMoeKSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_high_noise": ("MODEL", {"tooltip": "The first high-noise expert used for denoising the input latent."}),
                "model_high_noise_2": ("MODEL", {"tooltip": "The second high-noise expert used for denoising the input latent."}),
                "model_low_noise": ("MODEL", {"tooltip": "The low-noise expert used for denoising the input latent."}),
                "boundary": ("FLOAT", {"default": 0.875, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.001,"tooltip": "Boundary (or t_moe): Timestep at which models should be switched. Recommended values: 0.875 for t2v, 0.9 for i2v"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg_high_noise": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "CFG for the first high-noise model."}),
                "cfg_high_noise_2": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "CFG for the second high-noise model."}),
                "cfg_low_noise": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "CFG for the low-noise model."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The algorithm used when sampling."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler controls how noise is gradually removed."}),
                "sigma_shift": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.01}),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes to include."}),
                "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes to exclude."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "enable_second_high_pass": (["enable", "disable"], ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"

    CATEGORY = "sampling"
    DESCRIPTION = "Splits the high-noise region between two high models, then runs a single low model." 

    def sample(self, model_high_noise, model_high_noise_2, model_low_noise, boundary, seed, steps, cfg_high_noise, cfg_high_noise_2, cfg_low_noise, sampler_name, scheduler, sigma_shift, positive, negative, latent_image, enable_second_high_pass, denoise=1.0):
        model_high_noise = set_shift(model_high_noise, sigma_shift)
        model_high_noise_2 = set_shift(model_high_noise_2, sigma_shift)
        model_low_noise = set_shift(model_low_noise, sigma_shift)

        enable_flag = True
        if enable_second_high_pass == "disable":
            enable_flag = False

        # note cfgs order: first high, second high, low
        return wan_ksampler(model_high_noise, model_high_noise_2, model_low_noise, seed, steps, (cfg_high_noise, cfg_high_noise_2, cfg_low_noise), sampler_name, scheduler, positive, negative, latent_image, boundary=boundary, denoise=denoise, enable_second_high_pass=enable_flag)

class WanMoeKSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model_high_noise": ("MODEL", ),
                    "model_high_noise_2": ("MODEL", ),
                    "model_low_noise": ("MODEL", ),
                    "boundary": ("FLOAT", {"default": 0.875, "min": 0.0, "max": 1.0, "step": 0.001, "round":0.001}),
                    "add_noise": (["enable", "disable"], ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg_high_noise": ("FLOAT", {"default": 8.0}),
                    "cfg_high_noise_2": ("FLOAT", {"default": 8.0}),
                    "cfg_low_noise": ("FLOAT", {"default": 8.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "sigma_shift": ("FLOAT", {"default": 8.0}),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                    "enable_second_high_pass": (["enable", "disable"], ),
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, model_high_noise, model_high_noise_2, model_low_noise, boundary, add_noise, noise_seed, steps, cfg_high_noise, cfg_high_noise_2, cfg_low_noise, sampler_name, scheduler, sigma_shift, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, enable_second_high_pass, denoise=1.0):
        model_high_noise = set_shift(model_high_noise, sigma_shift)
        model_high_noise_2 = set_shift(model_high_noise_2, sigma_shift)
        model_low_noise = set_shift(model_low_noise, sigma_shift)
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True

        enable_flag = True
        if enable_second_high_pass == "disable":
            enable_flag = False

        return wan_ksampler(model_high_noise, model_high_noise_2, model_low_noise, noise_seed, steps, (cfg_high_noise, cfg_high_noise_2, cfg_low_noise), sampler_name, scheduler, positive, negative, latent_image, boundary=boundary, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise, enable_second_high_pass=enable_flag)

