# import logging
from dataclasses import dataclass
from typing import Optional, Literal

import accelerate

# import datasets
# import diffusers
# import transformers
# import torch

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,  # noqa
    UNet2DConditionModel,
)

# from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

# from diffusers.utils import check_min_version, deprecate, is_wandb_available
# from diffusers.utils.import_utils import is_xformers_available

from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

from spice_agent.utils.config import SPICE_MODEL_CACHE_FILEPATH, create_directory

logger = get_logger(__name__, log_level="INFO")


@dataclass
class SDPTConfig:
    # Model config
    pretrained_model_name_or_path: str
    revision: Optional[str] = None
    seed: Optional[int] = None
    use_ema: Optional[bool] = None
    non_ema_revision: Optional[str] = None

    # Dataset config
    # dataset_name: str
    # dataset_config_name: str
    # train_data_dir: str
    # image_column: str = "image"

    # Accelerator config
    output_dir: str = "sd-model-finetuned"
    logging_dir: str = "logs"
    gradient_accumulation_steps: int = 1
    mixed_precision: Optional[Literal["no", "fp16", "bf16"]] = None
    report_to: Optional[str] = None

    # input_perturbation: float = 0


@dataclass
class StableDiffusionModel:
    text_encoder: CLIPTextModel
    unet: UNet2DConditionModel
    vae: AutoencoderKL
    unet_ema: Optional[EMAModel] = None


class StableDiffusionTrainer:
    def __init__(self, config: SDPTConfig):
        self.config = config

        self.accelerator = self._get_accelerator()
        self.scheduler = self._get_scheduler()
        self.tokenizer = self._get_tokenizer()
        self.model = self._get_model()

        self.accelerator_output_dir = SPICE_MODEL_CACHE_FILEPATH.joinpath(
            self.config.output_dir
        )  # noqa
        self.accelerator_logging_dir = SPICE_MODEL_CACHE_FILEPATH.joinpath(
            self.config.logging_dir
        )  # noqa

        self._configure_trainer()

    def _configure_trainer(self):
        if self.config.seed is not None:
            set_seed(self.config.seed)

        self._configure_accelerator()

    # TODO: Document this
    def _get_accelerator(self) -> Accelerator:
        accelerator_project_config = ProjectConfiguration(
            project_dir=self.config.output_dir, logging_dir=self.config.logging_dir
        )
        accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            mixed_precision=self.config.mixed_precision,
            log_with=self.config.report_to,
            project_config=accelerator_project_config,
        )

        return accelerator

    def _configure_accelerator(self):
        unet_ema_dir = self.accelerator_output_dir.joinpath("unet_ema")
        unet_dir = self.accelerator_output_dir.joinpath("unet")

        if self.config.use_ema:
            create_directory(unet_ema_dir)

        create_directory(unet_dir)

        # create custom saving & loading hooks so that `accelerator.save_state(...)`
        # serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if self.config.use_ema and self.model.unet_ema:
                self.model.unet_ema.save_pretrained(unet_ema_dir)

            for i, model in enumerate(models):
                model.save_pretrained(unet_dir)

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if self.config.use_ema and self.model.unet_ema:
                load_model = EMAModel.from_pretrained(
                    unet_ema_dir, UNet2DConditionModel
                )
                self.model.unet_ema.load_state_dict(load_model.state_dict())
                self.model.unet_ema.to(self.accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(
                    input_dir, subfolder="unet"
                )
                model.register_to_config(**load_model.config)  # type: ignore

                model.load_state_dict(load_model.state_dict())  # type: ignore
                del load_model

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    # def _set_logging(self):
    #     # Make one log on every process with the configuration for debugging.
    #     logging.basicConfig(
    #         format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #         datefmt="%m/%d/%Y %H:%M:%S",
    #         level=logging.INFO,
    #     )
    #     if not self.accelerator:
    #         raise ValueError("Accelerator has not been configured!")

    #     logger.info(self.accelerator.state, main_process_only=False)

    #     if self.accelerator.is_local_main_process:
    #         datasets.utils.logging.set_verbosity_warning()
    #         transformers.utils.logging.set_verbosity_warning()
    #         diffusers.utils.logging.set_verbosity_info()
    #     else:
    #         datasets.utils.logging.set_verbosity_error()
    #         transformers.utils.logging.set_verbosity_error()
    #         diffusers.utils.logging.set_verbosity_error()
    #     pass

    def _get_scheduler(self) -> DDPMScheduler:
        scheduler = DDPMScheduler.from_pretrained(
            self.config.pretrained_model_name_or_path,  # type: ignore
            subfolder="scheduler",
        )

        return scheduler

    def _get_tokenizer(self) -> CLIPTokenizer:
        tokenizer = CLIPTokenizer.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=self.config.revision,
        )

        return tokenizer

    def _get_model(self) -> StableDiffusionModel:
        # TODO: In what context is deepspeed parameter sharding our unet -
        # is this useful and can it be translated to various unets/other
        # models across HF API?
        def deepspeed_zero_init_disabled_context_manager():
            """
            returns either a context list that includes one that will disable zero.
            Init or an empty context list
            """
            deepspeed_plugin = (
                AcceleratorState().deepspeed_plugin
                if accelerate.state.is_initialized()
                else None
            )
            if deepspeed_plugin is None:
                return []

            return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

        # TODO: Document this
        # Currently Accelerate doesn't know how to handle multiple models under
        # Deepspeed ZeRO stage 3. For this to work properly all models must be run
        # through `accelerate.prepare`. But accelerate
        # will try to assign the same optimizer with the same weights to all models
        # during `deepspeed.initialize`, which of course doesn't work.
        #
        # For now the following workaround will partially support Deepspeed ZeRO-3,
        # by excluding the 2 frozen models from being partitioned during `zero.Init`
        # which gets called during `from_pretrained` So CLIPTextModel and AutoencoderKL
        # will not enjoy the parameter sharding across multiple gpus and only
        # UNet2DConditionModel will get ZeRO sharded.
        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            text_encoder = CLIPTextModel.from_pretrained(
                self.config.pretrained_model_name_or_path,
                subfolder="text_encoder",
                revision=self.config.revision,
            )
            vae = AutoencoderKL.from_pretrained(
                self.config.pretrained_model_name_or_path,
                subfolder="vae",
                revision=self.config.revision,
            )

        unet = UNet2DConditionModel.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="unet",
            revision=self.config.non_ema_revision,
        )

        # Freeze vae and text_encoder
        vae.requires_grad_(False)  # type: ignore
        text_encoder.requires_grad_(False)

        model = StableDiffusionModel(
            text_encoder=text_encoder, unet=unet, vae=vae  # type: ignore
        )

        # Create EMA for the unet.
        if self.config.use_ema:
            unet_ema = EMAModel(
                unet.parameters(),  # type: ignore
                model_cls=UNet2DConditionModel,
                model_config=unet.config,  # type: ignore
            )
            model.unet_ema = unet_ema

        return model

        # TODO: Document this
        # if self.config.enable_xformers_memory_efficient_attention:
        #     if is_xformers_available():
        #         import xformers

        #         xformers_version = version.parse(xformers.__version__)
        #         if xformers_version == version.parse("0.0.16"):
        #             logger.warn(
        #                 "xFormers 0.0.16 cannot be used for training in some GPUs.
        #                   If you observe problems during training,
        #                   please update xFormers to at least 0.0.17.
        #                   See
        #   https://huggingface.co/docs/diffusers/main/en/optimization/xformers
        #                   for more details."
        #             )
        #         unet.enable_xformers_memory_efficient_attention()
        #     else:
        #         raise ValueError(
        #             "xformers is not available. Make sure it is installed correctly"
        #         )

    def compute_snr(self, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
            timesteps
        ].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
            device=timesteps.device
        )[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr


sdpt = StableDiffusionTrainer(
    SDPTConfig(pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4")
)

print(sdpt.model.unet)
