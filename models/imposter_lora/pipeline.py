"""
    modeled after the textual_inversion.py / train_dreambooth.py and the work
    of justinpinkney here: https://github.com/justinpinkney/stable-diffusion/blob/main/notebooks/imagic.ipynb
"""
import inspect
import warnings
from typing import List, Optional, Union

import numpy as np
import PIL
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.autograd import Variable

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers import DiffusionPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.utils import deprecate, logging
import torch.fft as fft 

# LORA Imports
from typing import Dict
#from diffusers.loaders import (
#    LoraLoaderMixin,
#    text_encoder_lora_state_dict,
#)
from diffusers.loaders import (
    LoraLoaderMixin,
)
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    SlicedAttnAddedKVProcessor,
)

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


class AnimationStableDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline for aerial diffusion.
    
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offsensive or harmful.
            Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        r"""
        Enable sliced attention computation.
        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.
        Args:
            slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                a number is provided, uses as many slices as `attention_head_dim // slice_size`. In this case,
                `attention_head_dim` must be a multiple of `slice_size`.
        """
        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        r"""
        Disable sliced attention computation. If `enable_attention_slicing` was previously invoked, this method will go
        back to computing attention in one step.
        """
        # set slice_size = `None` to disable `attention slicing`
        self.enable_attention_slicing(None)

    def train(
        self,
        prompt1: Union[str, List[str]],
        prompt2: Union[str, List[str]],
        prompt3: Union[str, List[str]],
        source_image: Union[torch.FloatTensor, PIL.Image.Image],
        driving_recon_image: Union[torch.FloatTensor, PIL.Image.Image],
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        generator: Optional[torch.Generator] = None,
        embedding_learning_rate: float = 0.001,
        diffusion_model_learning_rate: float = 2e-6,
        text_embedding_optimization_steps: int = 500,
        model_fine_tuning_optimization_steps: int = 1000,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `nd.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        #message = "Please use `image` instead of `init_image`."
        #init_image = deprecate("init_image", "0.14.0", message, take_from=kwargs)
        
        accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision="fp16",
        )

        if "torch_device" in kwargs:
            device = kwargs.pop("torch_device")
            warnings.warn(
                "`torch_device` is deprecated as an input argument to `__call__` and will be removed in v0.3.0."
                " Consider using `pipe.to(torch_device)` instead."
            )

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # Freeze vae and unet
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()

        if accelerator.is_main_process:
            accelerator.init_trackers(
                "imagic",
                config={
                    "embedding_learning_rate": embedding_learning_rate,
                    "text_embedding_optimization_steps": text_embedding_optimization_steps,
                },
            )

        self.prompt3 = prompt3 

        prompt = prompt1 
        # get text embeddings for prompt
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ) #Tokenizer output 77 dimensional, first word is at second position. 
        text_embeddings = torch.nn.Parameter(
            self.text_encoder(text_input.input_ids.to(self.device))[0], requires_grad=True
        ) # 77 x 768; 768 for each token
        text_embeddings = text_embeddings.detach()
        text_embeddings.requires_grad_()
        self.text_embeddings1 = text_embeddings
        
        prompt = prompt2 
        # get text embeddings for prompt
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ) #Tokenizer output 77 dimensional, first word is at second position. 
        text_embeddings2 = torch.nn.Parameter(
            self.text_encoder(text_input.input_ids.to(self.device))[0], requires_grad=True
        ) # 77 x 768; 768 for each token
        text_embeddings2 = text_embeddings2.detach()
        text_embeddings2.requires_grad_()
        self.text_embeddings2 = text_embeddings2
        
        # Initialize the optimizer
        optimizer = torch.optim.Adam(
            [text_embeddings],  # only optimize the embeddings
            lr=embedding_learning_rate,
        )


        if isinstance(source_image, PIL.Image.Image):
            source_image = preprocess(source_image)
        latents_dtype = text_embeddings.dtype        
        source_image = source_image.to(device=self.device, dtype=latents_dtype)
        init_latent_image_dist = self.vae.encode(source_image).latent_dist
        source_image_latents = init_latent_image_dist.sample(generator=generator)
        source_image_latents = 0.18215 * source_image_latents
        #self.source_fft = fft.fft2(source_image_latents, dim=(2,3), norm='ortho')
        self.source_fft = fft.fft2(source_image_latents, dim=(2,3))
        
        
        if isinstance(driving_recon_image, PIL.Image.Image):
            driving_recon_image = preprocess(driving_recon_image)
        driving_recon_image = driving_recon_image.to(device=self.device, dtype=latents_dtype)
        init_latent_image_dist = self.vae.encode(driving_recon_image).latent_dist
        driving_recon_image_latents = init_latent_image_dist.sample(generator=generator)
        driving_recon_image_latents = 0.18215 * driving_recon_image_latents
        #self.driving_fft = fft.fft2(driving_recon_image_latents, dim=(2,3), norm='ortho')
        self.driving_fft = fft.fft2(driving_recon_image_latents, dim=(2,3))
        
        progress_bar = tqdm(range(text_embedding_optimization_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        global_step = 0

        text_embeddings.requires_grad_(False)

        # Now we fine tune the unet to better reconstruct the image        
        # now we will add new LoRA weights to the attention layers
        device = "cuda"
        unet_lora_attn_procs = {}
        unet_lora_parameters = []
        for name, attn_processor in self.unet.attn_processors.items():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]

            if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
                lora_attn_processor_class = LoRAAttnAddedKVProcessor
            else:
                lora_attn_processor_class = (
                    LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
                )

            module = lora_attn_processor_class(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=4
            )
            module.to(device)
            unet_lora_attn_procs[name] = module
            unet_lora_parameters.extend(module.parameters())

        self.unet.set_attn_processor(unet_lora_attn_procs)
        #LORA Addition Done

        # Now we fine tune the unet to better reconstruct the image
        self.unet.requires_grad_(True)
        self.unet.train()
        optimizer = torch.optim.Adam(
            self.unet.parameters(),  # only optimize unet
            lr=diffusion_model_learning_rate,
        )
        progress_bar = tqdm(range(model_fine_tuning_optimization_steps), disable=not accelerator.is_local_main_process)

        logger.info("Next fine tuning the entire model to better reconstruct the init image")
        for _ in range(model_fine_tuning_optimization_steps):
            with accelerator.accumulate(self.unet.parameters()):
                # Sample noise that we'll add to the latents
                noise = torch.randn(source_image_latents.shape).to(source_image_latents.device)
                timesteps = torch.randint(1000, (1,), device=source_image_latents.device)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = self.scheduler.add_noise(source_image_latents, noise, timesteps)
                # Predict the noise residual
                noise_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample

                loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()

                noisy_latents = self.scheduler.add_noise(driving_recon_image_latents, noise, timesteps)
                noise_pred = self.unet(noisy_latents, timesteps, text_embeddings2).sample
                loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item()}  # , "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        accelerator.wait_for_everyone()          

    @torch.no_grad()
    def __call__(
        self,
        alpha: float = 1.2,
        prompt: Optional[str] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        K: float=12,
        amp_lr: float=1e-4,
        phase_lr: float=1.0,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `nd.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
        
        
        text_embeddings = self.text_embeddings1

        prompt = self.prompt3
        # get text embeddings for prompt
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ) #Tokenizer output 77 dimensional, first word is at second position. 
        text_embeddings = torch.nn.Parameter(
            self.text_encoder(text_input.input_ids.to(self.device))[0], requires_grad=True
        ) # 77 x 768; 768 for each token
        text_embeddings = text_embeddings.detach()
        self.text_embeddings3 = text_embeddings

        
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens = [""]
            max_length = self.tokenizer.model_max_length
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.view(1, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings1 = torch.cat([uncond_embeddings, self.text_embeddings1])
            text_embeddings2 = torch.cat([uncond_embeddings, self.text_embeddings2])
            text_embeddings3 = torch.cat([uncond_embeddings, self.text_embeddings3])
        
        
        # get the initial random noise unless the user supplied it

        # Unlike in other pipelines, latents need to be generated in the target device
        # for 1-to-1 results reproducibility with the CompVis implementation.
        # However this currently doesn't work in `mps`.
        latents_shape = (1, self.unet.in_channels, height // 8, width // 8)
        latents_dtype = text_embeddings.dtype
        if self.device.type == "mps":
            # randn does not exist on mps
            latents = torch.randn(latents_shape, generator=generator, device="cpu", dtype=latents_dtype).to(
                self.device
            )
        else:
            latents = torch.randn(latents_shape, generator=generator, device=self.device, dtype=latents_dtype)

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        #latents = self.gen_latent
            
        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            if(i<K):
                text_embeddings = text_embeddings2
            else:
                text_embeddings = text_embeddings3 

            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            phase_loss = (self.driving_fft.angle()).abs()
            phase_adj = fft.ifft2(torch.exp(phase_loss*1j) * 1, dim=(2,3)).abs()

            
            amp_loss = self.source_fft.abs()            
            amp_adj = fft.ifft2(amp_loss, dim=(2,3)).abs()

            if(i<49):
                latents.requires_grad_(True)
                optimizer = torch.optim.Adam(
                    [latents],  # only optimize the embeddings
                    lr=1.0,
                )
                generated_fft = fft.fft2(latents, dim=(2,3))
                loss1 = torch.mean((generated_fft.angle().abs()-phase_loss)**2)
                loss2 = torch.mean((amp_loss-generated_fft.abs())**2)
                print(loss1, loss2)
                loss = phase_lr * loss1 + amp_lr * loss2 
                loss = Variable(loss, requires_grad = True)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                latents.requires_grad_(False)
            
            #latents = latents - phase_adj - 1e-4 * amp_adj  
            

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
 
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(
                self.device
            )
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(text_embeddings.dtype)
            )
        else:
            has_nsfw_concept = None

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)