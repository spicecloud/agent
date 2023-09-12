import torch
import logging
import cv2
import numpy as np
import PIL.Image

from typing import Any, List, Union, Type, Tuple
from transformers import AutoImageProcessor, DPTForDepthEstimation

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
)

from spice_agent.inference.tasks import (
    StableDiffusionPipelineTask,
    StableDiffusionImg2ImgPipelineTask,
    StableDiffusionXLPipelineTask,
    StableDiffusionXLImg2ImgPipelineTask,
    StableDiffusionXLControlNetPipelineTask,
)

from spice_agent.inference.types import (
    SUPPORTED_MODEL_REPOSITORY_IDS,
    StableDiffusionPipelineTaskState,
    TStableDiffusionPipelineRunConfig,
    StableDiffusionPipelineRunConfig,
    StableDiffusionImg2ImgPipelineRunConfig,
    StableDiffusionXLPipelineRunConfig,
    StableDiffusionXLImg2ImgPipelineRunConfig,
    StableDiffusionXLControlNetPipelineRunConfig,
)

LOGGER = logging.getLogger(__name__)


class StableDiffusionTaskFactory:
    def __init__(self):
        self.variant = "fp16"
        self.torch_dtype = torch.float16
        if torch.backends.mps.is_available():
            self.variant = "fp32"
            self.torch_dtype = torch.float32
        self.use_safetensors = True
        self.pipeline_state = StableDiffusionPipelineTaskState()

    def create_task(
        self,
        device: Union[int, str, "torch.device"],
        model_repo_id: str,
        task_config: dict[str, Any],
        inference_options: dict[str, Any],
    ):
        """
        Generates a type of stable diffusion pipeline task depending on
        the model repo id
        """

        if model_repo_id not in SUPPORTED_MODEL_REPOSITORY_IDS:
            message = f"{model_repo_id} is not a supported inference model!"
            LOGGER.error(message)
            raise ValueError(message)

        if model_repo_id == "stabilityai/stable-diffusion-2-1-base":
            return self._create_stable_diffusion_pipeline_task(
                device, model_repo_id, task_config, inference_options
            )
        elif model_repo_id == "stabilityai/stable-diffusion-xl-base-1.0":
            return self._create_stable_diffusion_xl_pipeline_task(
                device, model_repo_id, task_config, inference_options
            )

    def configure_ensemble_of_denoisers(
        self,
        base_task: StableDiffusionXLPipelineTask,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] = "",
    ) -> Tuple[StableDiffusionXLPipelineTask, StableDiffusionXLImg2ImgPipelineTask]:
        """
        Reconfigures base_task to serve as the expert for the high-noise diffusion stage
        and generates a refinement task for the low-noise diffusion stage.

        Since we have to wait for the result of base_task run_pipeline()
        to complete, the following should be added before refiner and after base_task:
        refinement_task.pipeline_run_config.image = base_task.run_pipeline().output.images

        Returns (StableDiffusionXLPipelineTask, StableDiffusionXLImg2ImgPipelineTask)
        """  # noqa
        if not isinstance(base_task, StableDiffusionXLPipelineTask):
            message = f"{type(base_task)} cannot be configured with a refinement step!"
            LOGGER.error(message)
            raise ValueError(message)

        # Reconfigure base_task for high-noise diffusion stage
        base_task.pipeline_run_config.output_type = "latent"
        base_task.pipeline_run_config.denoising_end = 0.8

        # Setup low-noise diffusion stage
        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=base_task.pipeline.text_encoder_2,
            vae=base_task.pipeline.vae,
            torch_dtype=self.torch_dtype,
            variant=self.variant,
            use_safetensors=True,
        )

        refiner_state = StableDiffusionPipelineTaskState()
        refiner_run_config = StableDiffusionXLImg2ImgPipelineRunConfig(
            prompt=prompt, negative_prompt=negative_prompt
        )
        refiner_run_config.denoising_start = 0.8

        refinement_task = StableDiffusionXLImg2ImgPipelineTask(
            device=base_task.device,
            pipeline=refiner,
            pipeline_state=refiner_state,
            pipeline_run_config=refiner_run_config,
            # Turns off image preview since the refinement step is fast
            enable_image_preview=False,
        )

        return (base_task, refinement_task)

    def _prepare_pipeline_run_config(
        self,
        options: dict[str, Any],
        type_config: Type[TStableDiffusionPipelineRunConfig],
    ) -> TStableDiffusionPipelineRunConfig:
        """
        Filters options into a dictionary that has the same keys as config
        and returns an instance of TStableDiffusionPipelineRunConfig
        with the filtered options.
        """
        filtered_options = {
            key: value
            for key, value in options.items()
            if key in type_config.__annotations__
        }

        return type_config(**filtered_options)

    def _create_stable_diffusion_pipeline_task(
        self,
        device: Union[int, str, "torch.device"],
        model_repo_id: str,
        task_config: dict[str, Any],
        inference_options: dict[str, Any],
    ):
        is_file_input_paths = task_config.get("is_file_input_paths", False)

        if not is_file_input_paths:
            # pass pipeline_args, device, and options
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_repo_id,
                torch_dtype=self.torch_dtype,
                variant=self.variant,
                use_safetensors=self.use_safetensors,
            )
            pipeline_run_config = self._prepare_pipeline_run_config(
                inference_options, StableDiffusionPipelineRunConfig
            )
            return StableDiffusionPipelineTask(
                device=device,
                pipeline=pipeline,
                pipeline_state=self.pipeline_state,
                pipeline_run_config=pipeline_run_config,
            )
        else:
            pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_repo_id,
                torch_dtype=self.torch_dtype,
                variant=self.variant,
                use_safetensors=self.use_safetensors,
            )
            pipeline_run_config = self._prepare_pipeline_run_config(
                inference_options, StableDiffusionImg2ImgPipelineRunConfig
            )
            return StableDiffusionImg2ImgPipelineTask(
                device=device,
                pipeline=pipeline,
                pipeline_state=self.pipeline_state,
                pipeline_run_config=pipeline_run_config,
            )

    def _get_control_nets(self, control_types: List[str]) -> dict[str, ControlNetModel]:
        control_nets: dict[str, ControlNetModel] = {}
        for control_type in control_types:
            type = control_type.lower()
            if type == "canny":
                control_nets["canny"] = ControlNetModel.from_pretrained(  # type: ignore
                    "diffusers/controlnet-canny-sdxl-1.0",
                    torch_dtype=self.torch_dtype,
                    variant=self.variant,
                    use_safetensors=self.use_safetensors,
                )
            if type == "depth":
                control_nets["depth"] = ControlNetModel.from_pretrained(  # type: ignore
                    "diffusers/controlnet-depth-sdxl-1.0",
                    torch_dtype=self.torch_dtype,
                    variant=self.variant,
                    use_safetensors=self.use_safetensors,
                )
            if type != "canny" and type != "depth":
                message = f"{control_type} is unknown"
                LOGGER.warn(message)

        return control_nets

    def _apply_depth_transform(self, device, images):
        depth_images: List[PIL.Image.Image] = []
        image_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
        depth_estimator = DPTForDepthEstimation.from_pretrained(
            "Intel/dpt-hybrid-midas"
        )

        for image in images:
            # Preprocess images for depth estimator
            inputs = image_processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = depth_estimator(**inputs)  # type: ignore
                predicted_depth = outputs.predicted_depth

            # interpolate to original size
            depth_map = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.size[::-1],  # grabs size of image
                mode="bicubic",
                align_corners=False,
            )

            depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
            depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
            depth_image = torch.cat([depth_map] * 3, dim=1)

            depth_image = depth_image.permute(0, 2, 3, 1).cpu().numpy()[0]
            depth_image = PIL.Image.fromarray(
                (depth_image * 255.0).clip(0, 255).astype(np.uint8)
            )
            depth_images.append(depth_image)

        return depth_images

    def _apply_canny_transform(self, images):
        canny_images: List[PIL.Image.Image] = []
        for image in images:
            image = np.array(image)
            image = cv2.Canny(image, 100, 200)
            image = image[:, :, None]
            image = np.concatenate([image, image, image], axis=2)
            canny_images.append(PIL.Image.fromarray(image))
        return canny_images

    def _compute_conditioned_images(self, device, images, control_net_types: List[str]):
        conditioned_images = []
        for type in control_net_types:
            if type == "canny":
                conditioned_images.extend(self._apply_canny_transform(images))
            if type == "depth":
                conditioned_images.extend(self._apply_depth_transform(device, images))
        return conditioned_images

    def _create_stable_diffusion_xl_pipeline_task(
        self,
        device: Union[int, str, "torch.device"],
        model_repo_id: str,
        task_config: dict[str, Any],
        inference_options: dict[str, Any],
    ):
        is_file_input_paths = task_config.get("is_file_input_paths", False)
        is_control = task_config.get("is_control", False)
        # is_mask = task_config.get("is_mask", False)
        control_types = task_config.get("control_types", [])

        if not is_file_input_paths:
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_repo_id,
                torch_dtype=self.torch_dtype,
                variant=self.variant,
                use_safetensors=self.use_safetensors,
            )
            pipeline_run_config = self._prepare_pipeline_run_config(
                inference_options, StableDiffusionXLPipelineRunConfig
            )
            return StableDiffusionXLPipelineTask(
                device=device,
                pipeline=pipeline,
                pipeline_state=self.pipeline_state,
                pipeline_run_config=pipeline_run_config,
            )
        else:
            if is_control and control_types:
                control_nets_dict = self._get_control_nets(control_types)
                control_net_types = list(control_nets_dict.keys())
                control_nets = list(control_nets_dict.values())

                images = self._compute_conditioned_images(
                    device, inference_options["image"], control_net_types
                )
                inference_options["image"] = images

                pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                    model_repo_id,
                    torch_dtype=self.torch_dtype,
                    variant=self.variant,
                    use_safetensors=self.use_safetensors,
                    controlnet=control_nets,
                )
                pipeline_run_config = self._prepare_pipeline_run_config(
                    inference_options, StableDiffusionXLControlNetPipelineRunConfig
                )
                return StableDiffusionXLControlNetPipelineTask(
                    device=device,
                    pipeline=pipeline,
                    pipeline_state=self.pipeline_state,
                    pipeline_run_config=pipeline_run_config,
                )
            else:
                pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-refiner-1.0",
                    torch_dtype=self.torch_dtype,
                    variant=self.variant,
                    use_safetensors=self.use_safetensors,
                )
                pipeline_run_config = self._prepare_pipeline_run_config(
                    inference_options, StableDiffusionXLImg2ImgPipelineRunConfig
                )
                return StableDiffusionXLImg2ImgPipelineTask(
                    device=device,
                    pipeline=pipeline,
                    pipeline_state=self.pipeline_state,
                    pipeline_run_config=pipeline_run_config,
                )
