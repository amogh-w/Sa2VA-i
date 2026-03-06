# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import warnings
from typing import Any, List, Optional, Tuple, Union

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

import torch.utils.checkpoint
import transformers

from .modeling_internlm2 import InternLM2ForCausalLM
from .modeling_phi3 import Phi3ForCausalLM
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModel,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    Qwen2ForCausalLM,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
from transformers import StoppingCriteriaList, StoppingCriteria

from .configuration_sa2va_chat import Sa2VAChatConfig
from .modeling_intern_vit import InternVisionModel, has_flash_attn

from .sam2 import SAM2
from .templates import PROMPT_TEMPLATE

import numpy as np
from torchvision.transforms.functional import resize, to_pil_image

from types import MethodType
import torch.nn.functional as F

try:
    from .flash_attention import FlashAttention

    has_flash_attn = True
except:
    print("FlashAttention is not installed.")
    has_flash_attn = False

logger = logging.get_logger(__name__)

"""
# --------------------------
# Helper Functions and Classes
# --------------------------
"""

DEBUG = True

from termcolor import cprint


def debug_info(*args):
    if DEBUG:
        cprint(f"[INFO] {' '.join(map(str, args))}", "yellow")


def debug_error(*args):
    if DEBUG:
        cprint(f"[ERROR] {' '.join(map(str, args))}", "red")


def debug_success(*args):
    if DEBUG:
        cprint(f"[SUCCESS] {' '.join(map(str, args))}", "green")


# Compares library versions
def version_cmp(v1, v2, op="eq"):
    import operator
    from packaging import version

    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


# Rule that tells the LLM to stop generating text if it outputs a specific word
class StopWordStoppingCriteria(StoppingCriteria):
    """StopWord stopping criteria."""

    def __init__(self, tokenizer, stop_word):
        self.tokenizer = tokenizer
        self.stop_word = stop_word
        self.length = len(self.stop_word)

    def __call__(self, input_ids, *args, **kwargs) -> bool:
        cur_text = self.tokenizer.decode(input_ids[0])
        cur_text = cur_text.replace("\r", "").replace("\n", "")
        return cur_text[-self.length :] == self.stop_word


def get_stop_criteria(
    tokenizer,
    stop_words=[],
):
    stop_criteria = StoppingCriteriaList()
    for word in stop_words:
        stop_criteria.append(StopWordStoppingCriteria(tokenizer, word))
    return stop_criteria


# Utility to resize images to a fixed square size (default 1024x1024) for the segmentation model
class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode="RGB")
        return np.array(img.resize((self.target_length, self.target_length)))


# Tiling strategy.
# Breaks a large or high-aspect-ratio image into smaller blocks (tiles) so the model can see fine details without losing resolution
def dynamic_preprocess(
    image, min_num=1, max_num=6, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


from transformers.cache_utils import Cache, DynamicCache


def prepare_inputs_for_generation_phi3(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    **kwargs,
):
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_max_length()
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]
            max_cache_length = None

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and (
        past_key_values is None or len(past_key_values) == 0
    ):
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs


"""
# --------------------------
# Core Model
# --------------------------
"""


# Main Class
# Inheriting from PreTrainedModel.
# It manages three main components:
#   Vision Model: An InternVisionModel that "sees" the image.
#   Language Model: Can be Llama, InternLM2, Phi3, or Qwen2.
#   Grounding Encoder: A SAM2 model used to generate segmentation masks.
class Sa2VAChatModel(PreTrainedModel):
    config_class = Sa2VAChatConfig
    main_input_name = "pixel_values"
    base_model_prefix = "language_model"
    _no_split_modules = [
        "InternVisionModel",
        "LlamaDecoderLayer",
        "InternLM2DecoderLayer",
        "Phi3DecoderLayer",
        "Qwen2DecoderLayer",
        "SAM2",
    ]
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True

    # Initializes the architectures and sets up the Projector (MLP1)
    def __init__(
        self,
        config: Sa2VAChatConfig,
        vision_model=None,
        language_model=None,
        use_flash_attn=True,
    ):
        super().__init__(config)

        assert version_cmp(transformers.__version__, "4.37.0", "ge")
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.template = self.template.replace("-", "_")
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (config.downsample_ratio**2)
        )
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.llm_arch_name = config.llm_config.architectures[0]

        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config._attn_implementation = (
            "flash_attention_2" if use_flash_attn else "eager"
        )

        logger.info(f"num_image_token: {self.num_image_token}")
        logger.info(f"ps_version: {self.ps_version}")
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == "LlamaForCausalLM":
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == "InternLM2ForCausalLM":
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == "Phi3ForCausalLM":
                self.language_model = Phi3ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == "Qwen2ForCausalLM":
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(
                    f"{config.llm_config.architectures[0]} is not implemented."
                )

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(
                vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size
            ),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

        self.img_context_token_id = None
        self.conv_template = PROMPT_TEMPLATE[self.template]
        self.template = self.conv_template
        if hasattr(config, "system_message"):
            self.system_message = config.system_message
        self.num_samples = 0

        if config.use_backbone_lora:
            self.wrap_backbone_lora(
                r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora
            )

        if config.use_llm_lora:
            self.wrap_llm_lora(
                r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora
            )

        self.grounding_encoder = SAM2()
        out_dim = self.grounding_encoder.hidden_dim
        in_dim = llm_hidden_size
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        )

        self.init_prediction_config = False

    # Allows the model to be fine-tuned efficiently by only updating a tiny fraction of the parameters
    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        # Determine the target modules based on the architecture of the language model
        if self.llm_arch_name == "InternLM2ForCausalLM":
            target_modules = [
                "attention.wqkv",
                "attention.wo",
                "feed_forward.w1",
                "feed_forward.w2",
                "feed_forward.w3",
            ]
        elif self.llm_arch_name == "Phi3ForCausalLM":
            target_modules = [
                "mlp.down_proj",
                "mlp.gate_up_proj",
                "self_attn.o_proj",
                "self_attn.qkv_proj",
            ]
        elif self.llm_arch_name in ["Qwen2ForCausalLM", "LlamaForCausalLM"]:
            target_modules = [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.down_proj",
                "mlp.up_proj",
            ]
        else:
            raise NotImplemented
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type="CAUSAL_LM",
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()

    # Technique used to downsample image features spatially while increasing the depth of the information
    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        if self.ps_version == "v1":
            warnings.warn(
                "In ps_version 'v1', the height and width have not been swapped back, "
                "which results in a transposed image."
            )
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    # Vision Model → Pixel Shuffle → MLP Projector.
    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=False, return_dict=True
            ).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True
            ).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    # This is the main entry point for the model during the training phase.
    # It calculates the "loss" (how far off the prediction was) to help the model learn.
    def forward(self, data, data_samples=None, mode="loss"):
        # Extract the raw pixel data (images/video frames) from the input dictionary
        pixel_values = data["pixel_values"]

        # Check if the input is a list of images or a 5D tensor (usually [Batch, Frames, Channels, H, W])
        if type(pixel_values) is list or pixel_values.ndim == 5:
            # If it's a list, ensure every image has a batch dimension (4D: [1, C, H, W])
            if type(pixel_values) is list:
                pixel_values = [
                    x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                ]

            # Combine all images/frames into a single large continuous block of memory (tensor)
            # We also ensure the pixels match the precision (dtype) of the vision model
            concat_images = torch.cat(
                [image.to(self.vision_model.dtype) for image in pixel_values], dim=0
            )
        else:
            # If the data format isn't a list or 5D tensor, the code stops here
            raise NotImplementedError()

        # Extract standard language model inputs: token IDs, positions, and the padding mask
        input_ids = data["input_ids"]
        position_ids = data["position_ids"]
        attention_mask = data["attention_mask"]

        # Determine which items in the batch are actual images vs empty/padding
        # torch.sum(dim=(1,2,3)) checks if there is any color data in the pixels
        # If the sum is 0, the flag is False; if it has content, the flag is True
        image_flags = torch.sum(concat_images, dim=(1, 2, 3)) != 0
        # Convert True/False to 1/0 for numerical processing
        image_flags = image_flags.long()

        # 'labels' are the ground-truth answers the model is supposed to predict
        labels = data["labels"]
        # During training, we don't use KV-cache (that's for fast generation only)
        use_cache = False

        # Visual Prompting: Check if there is a mask indicating which image to focus on
        if "vp_overall_mask" not in data.keys():
            vp_overall_mask = None
        else:
            vp_overall_mask = data["vp_overall_mask"]

        # Segmentation: Check if there are specific pixel-level masks for object grounding
        if "prompt_masks" in data.keys():
            prompt_masks = data["prompt_masks"]
        else:
            prompt_masks = None

        # Pass all the organized data to the specialized LLM forward function.
        # This is where the images and text are physically merged together.
        outputs = self._llm_forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            pixel_values=concat_images,
            labels=labels,
            use_cache=use_cache,
            output_hidden_states=True,  # We need hidden states for segmentation tasks
            vp_overall_mask=vp_overall_mask,
            prompt_masks=prompt_masks,
        )

        # Return the final loss and logit predictions
        return outputs

    def _llm_forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        vp_overall_mask=None,
        prompt_masks=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # Determine if we should return a dictionary or a standard tuple
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Remove extra dimension from image_flags (indicates which batch entries have images)
        image_flags = image_flags.squeeze(-1)

        # Step 1: Convert input_ids into text embeddings using the LLM's embedding layer
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        # Step 2: Extract visual features from the pixels using the Vision Transformer (ViT)
        vit_embeds = self.extract_feature(pixel_values)
        # Ensure visual features match the precision (e.g., bfloat16) of the text embeddings
        vit_embeds = vit_embeds.to(input_embeds.dtype)
        fast_vit_embeds = None

        # Filter vit_embeds to only include samples where image_flags indicate an actual image exists
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        # Flatten the batch (B) and sequence length (N) dimensions for easy indexing
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        self._count += 1

        # Step 3: Visual Prompt (VP) Logic - Handles specific regions of the image
        if vp_overall_mask is not None and prompt_masks is not None:
            vp_embeds = []
            vp_overall_mask = vp_overall_mask.to(vit_embeds.device).bool()
            prompt_masks = [item.to(vit_embeds.device).bool() for item in prompt_masks]

            # Identify which images in the batch have specific regional masks
            vp_overall_mask = vp_overall_mask[image_flags == 1]
            overall_tile_vit_embeds = vit_embeds[vp_overall_mask]  # (n_img, hw, c)

            i_vp_img = 0
            for i_img in range(len(vit_embeds)):
                # Add the full image features first
                vp_embeds.append(vit_embeds[i_img].reshape(-1, C))
                # If this image has a "Visual Prompt" (region mask), extract those specific features
                if vp_overall_mask[i_img]:
                    tile_vit_embeds = overall_tile_vit_embeds[i_vp_img].reshape(-1, C)
                    objects_prompt_masks = prompt_masks[i_vp_img]
                    n_obj = len(objects_prompt_masks)
                    # Expand the visual features for each object mask detected
                    tile_vit_embeds = tile_vit_embeds.unsqueeze(0).repeat(n_obj, 1, 1)
                    objects_prompt_masks = objects_prompt_masks.reshape(n_obj, -1)
                    # Only keep the visual features covered by the mask (the "region")
                    vp_embeds.append(tile_vit_embeds[objects_prompt_masks])
                    i_vp_img += 1
            # Combine all visual features (full images + regions) into one tensor
            vp_embeds = torch.cat(vp_embeds, dim=0)
        else:
            vp_embeds = None

        # Step 4: Substitution - Replace placeholder tokens with visual features
        input_ids = input_ids.reshape(B * N)
        # Find every index where the text contains the special "<IMG_CONTEXT>" token
        selected = input_ids == self.img_context_token_id

        if vp_embeds is None:
            # Standard path: Replace placeholders with the extracted ViT features
            try:
                input_embeds[selected] = vit_embeds.reshape(-1, C)
            except Exception as e:
                # Safety fallback: if token count mismatches, expand/truncate features to fit
                vit_embeds = vit_embeds.reshape(-1, C)
                n_token = selected.sum()
                if n_token > len(vit_embeds):
                    expand_ratio = n_token // len(vit_embeds) + 1
                    vit_embeds = torch.cat([vit_embeds] * expand_ratio, dim=0)
                input_embeds[selected] = vit_embeds[:n_token]
        else:
            # Regional path: Replace placeholders with the combined image + region features
            try:
                input_embeds[selected] = vp_embeds.reshape(-1, C)
            except Exception as e:
                vp_embeds = vp_embeds.reshape(-1, C)
                n_token = selected.sum()
                if n_token > len(vp_embeds):
                    expand_ratio = n_token // len(vp_embeds) + 1
                    vp_embeds = torch.cat([vp_embeds] * expand_ratio, dim=0)
                input_embeds[selected] = vp_embeds[:n_token]

        # Reshape the embeddings back to (Batch, Sequence, Channels) for the LLM
        input_embeds = input_embeds.reshape(B, N, C)

        # Step 5: LLM Inference - Feed the merged vision-text sequence into the language model
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        # Step 6: Loss Calculation (for Training)
        loss = None
        if labels is not None:
            # Shift sequences: text at index 'i' should predict token at index 'i+1'
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Standard CrossEntropyLoss to compare predicted word probabilities vs actual word
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        # Return the results in the requested format
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()  # Disable gradient calculation to save memory and speed up inference
    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        prompt_masks=None,
        vp_overall_mask=None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        device = self.device
        # Ensure the model knows which ID corresponds to the image placeholder
        assert self.img_context_token_id is not None

        if pixel_values is not None:
            # If features are pre-calculated, use them; otherwise, extract them now
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                # Handle multi-image or video frames by stacking them into a 4D batch
                if type(pixel_values) is list or pixel_values.ndim == 5:
                    if type(pixel_values) is list:
                        pixel_values = [
                            x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                        ]
                    pixel_values = torch.cat(
                        [image.to(self.vision_model.dtype) for image in pixel_values],
                        dim=0,
                    )

                # Convert raw pixels into visual feature vectors (embeddings)
                vit_embeds = self.extract_feature(pixel_values.to(device))

            # Identify valid images (non-zero pixels)
            image_flags = torch.sum(pixel_values, dim=(1, 2, 3)) != 0
            image_flags = image_flags.long()
            vit_embeds = vit_embeds[image_flags == 1]

            # Convert text input_ids into text embeddings
            input_embeds = self.language_model.get_input_embeddings()(
                input_ids.to(device)
            )
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            # --- Visual Prompt (VP) Logic: Handles regional/masked area features ---
            if vp_overall_mask is not None and prompt_masks is not None:
                vp_embeds = []
                vp_overall_mask = vp_overall_mask.to(vit_embeds.device).bool()
                prompt_masks = [
                    item.to(vit_embeds.device).bool() for item in prompt_masks
                ]

                vp_overall_mask = vp_overall_mask[image_flags == 1]
                overall_tile_vit_embeds = vit_embeds[vp_overall_mask]

                for i_img in range(len(vit_embeds)):
                    vp_embeds.append(vit_embeds[i_img].reshape(-1, C))  # Add base image
                    if vp_overall_mask[i_img]:
                        # Add specific regional features for masks (Visual Prompts)
                        tile_vit_embeds = overall_tile_vit_embeds[i_vp_img].reshape(
                            -1, C
                        )
                        objects_prompt_masks = prompt_masks[i_vp_img]
                        n_obj = len(objects_prompt_masks)
                        tile_vit_embeds = tile_vit_embeds.unsqueeze(0).repeat(
                            n_obj, 1, 1
                        )
                        objects_prompt_masks = objects_prompt_masks.reshape(n_obj, -1)
                        vp_embeds.append(tile_vit_embeds[objects_prompt_masks])
                        i_vp_img += 1

                vp_embeds = torch.cat(vp_embeds, dim=0)
            else:
                vp_embeds = None

            # --- Embedding Substitution: Replace tokens with visual data ---
            input_ids = input_ids.reshape(B * N)
            selected = (
                input_ids == self.img_context_token_id
            )  # Find context token locations
            assert selected.sum() != 0

            if vp_embeds is None:
                # Replace <IMG_CONTEXT> tokens with full image features
                input_embeds[selected] = vit_embeds.reshape(-1, C).to(
                    input_embeds.device
                )
            else:
                # Replace <IMG_CONTEXT> tokens with combined Image + Region features
                # Includes safety check for token count mismatch
                if len(input_embeds[selected]) != len(vp_embeds.reshape(-1, C)):
                    min_tokens = min(
                        len(input_embeds[selected]), len(vp_embeds.reshape(-1, C))
                    )
                    input_embeds[selected][:min_tokens] = vp_embeds.reshape(-1, C)[
                        :min_tokens
                    ].to(input_embeds.device)
                else:
                    input_embeds[selected] = vp_embeds.reshape(-1, C).to(
                        input_embeds.device
                    )

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            # Pure text fallback if no images are provided
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Trigger the LLM's autoregressive generation process
        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask.to(device),
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
            use_cache=True,  # Use KV-caching for faster text generation
            **generate_kwargs,
        )

        return outputs

    def preparing_for_generation(
        self, tokenizer, max_new_tokens=2048, torch_dtype=torch.bfloat16
    ):
        # Attach the tokenizer to the model instance
        if not hasattr(self, "tokenizer"):
            self.tokenizer = tokenizer

        self.bot_name = "BOT"

        # --- Stop Criteria Setup ---
        # Fetch stop words from the template and build the "brakes" for text generation
        stop_words = []
        stop_words += self.template.get("STOP_WORDS", [])
        stop_criteria = get_stop_criteria(
            tokenizer=self.tokenizer, stop_words=stop_words
        )
        self.stop_criteria = stop_criteria

        # --- Generation Configuration ---
        # Define how the LLM should behave (length, sampling, padding)
        default_generation_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Use greedy search for more stable answers
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id),
        )

        self.gen_config = GenerationConfig(**default_generation_kwargs)
        self.init_prediction_config = True
        self.torch_dtype = torch_dtype
        self.to(torch_dtype)  # Move model to correct precision (e.g., bfloat16)

        # --- Image Processing Constants ---
        self.extra_image_processor = DirectResize(
            target_length=1024
        )  # For SAM2 grounding
        self.min_dynamic_patch = 1
        self.max_dynamic_patch = 12
        self.downsample_ratio = 0.5
        self.image_size = 448
        self.use_thumbnail = True
        patch_size = 14
        self.patch_size = patch_size

        # Calculate how many tokens one image occupies in the LLM's "brain"
        self.patch_token = int(
            (self.image_size // patch_size) ** 2 * (self.downsample_ratio**2)
        )

        # Standard ImageNet normalization values
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)

        # Define the string markers used in the text prompt
        self.IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        self.IMG_START_TOKEN = "<img>"
        self.IMG_END_TOKEN = "</img>"

        # --- Vision Transformation Pipeline ---
        # Prepares raw PIL images for the Vision Transformer
        self.transformer = T.Compose(
            [
                T.Lambda(
                    lambda img: img.convert("RGB") if img.mode != "RGB" else img
                ),  # Ensure RGB
                T.Resize(
                    (self.image_size, self.image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                T.ToTensor(),
                T.Normalize(
                    mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD
                ),  # Normalize colors
            ]
        )

        self.VP_START_TOKEN = "<vp>"
        self.VP_END_TOKEN = "</vp>"

        # Patch Phi3 specific logic if that architecture is being used
        if self.config.llm_config.architectures[0] == "Phi3ForCausalLM":
            self.language_model.prepare_inputs_for_generation = MethodType(
                prepare_inputs_for_generation_phi3, self.language_model
            )

        # Store the numeric IDs of critical tokens for fast lookup during generation
        self.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.seg_token_idx = tokenizer.convert_tokens_to_ids("[SEG]")
        return

    # --------------------------
    # Unified Multi-Modality Processing
    # --------------------------

    def _process_visual_input(self, visual_data, sample_idx=None, mask_prompts=None):
        """
        Unified processor for Images, Videos, and Multi-Image inputs.
        """
        # 1. Normalize input to a list of images
        images = visual_data if isinstance(visual_data, list) else [visual_data]
        if sample_idx is None:
            sample_idx = list(range(len(images)))

        all_pixel_values = []
        image_token_str = ""
        vp_token_str = ""
        vp_overall_mask = None
        processed_mask_prompts = None

        # 2. THE ViT PATH: Prepare pixels for LLM understanding
        debug_info(f"Processing ViT path for {len(sample_idx)} sampled frames")
        for i in sample_idx:
            img = images[i]
            # Spatial tiling: splits high-res images into manageable patches
            tiles = dynamic_preprocess(
                img, self.min_dynamic_patch, self.max_dynamic_patch,
                self.image_size, self.use_thumbnail,
            )
            
            # Convert patches to normalized tensors
            pixel_val = torch.stack([self.transformer(t) for t in tiles]).to(self.torch_dtype)
            all_pixel_values.append(pixel_val)

            # Build the text string: <IMG_START><IMG_CONTEXT * N><IMG_END>
            num_tokens = pixel_val.shape[0] * self.patch_token
            image_token_str += f"{self.IMG_START_TOKEN}{self.IMG_CONTEXT_TOKEN * num_tokens}{self.IMG_END_TOKEN}\n"

        pixel_values = torch.cat(all_pixel_values, dim=0)
        debug_info(f"ViT pixel_values shape: {pixel_values.shape}")

        # 3. THE SAM2 PATH: Prepare pixels for Grounding/Segmentation
        g_indices = range(len(images)) if len(images) > 1 else [0]
        extra_pixel_values = []
        
        debug_info(f"Processing SAM2 path for {len(g_indices)} frames")
        for idx in g_indices:
            g_image = np.array(images[idx])
            g_image = self.extra_image_processor.apply_image(g_image)
            extra_pixel_values.append(
                torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
            )

        g_pixel_values = torch.stack(
            [self.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values]
        ).to(self.torch_dtype)
        debug_info(f"SAM2 g_pixel_values shape: {g_pixel_values.shape}")

        # 4. VISUAL PROMPTS (VP): Handling user-defined region masks
        if mask_prompts is not None:
            if len(images) == 1:
                debug_info(f"Processing {len(mask_prompts[0])} mask prompts")
                # Mask applied to the last tile (the thumbnail/global view)
                vp_overall_mask = torch.Tensor([False] * (len(all_pixel_values[0]) - 1) + [True])
                
                mask_prompts_t = [torch.Tensor(item).to(pixel_values.device) for item in mask_prompts]
                
                # Resize masks to match internal feature map resolution
                target_res = int(self.image_size // self.patch_size * self.downsample_ratio)
                processed_mask_prompts = [
                    F.interpolate(item.unsqueeze(0), size=(target_res, target_res), mode="nearest").squeeze(0)
                    for item in mask_prompts_t
                ]

                # Generate region descriptions for the prompt
                region_pixels = [m.bool().to(torch.int64).sum() for m in processed_mask_prompts[0]]
                vp_token_str = f"\nThere are {len(processed_mask_prompts[0])} part regions in the picture: "
                for i, count in enumerate(region_pixels):
                    vp_token_str += f"region{i + 1}{self.VP_START_TOKEN}{self.IMG_CONTEXT_TOKEN * count}{self.VP_END_TOKEN}"
                    vp_token_str += ".\n" if i == len(region_pixels) - 1 else ", "
            else:
                debug_error("Mask prompts provided but input is not a single image. Skipping VP.")

        return (
            pixel_values, 
            g_pixel_values, 
            (image_token_str + vp_token_str).strip(), 
            vp_overall_mask, 
            processed_mask_prompts
        )

    def _decode_masks(self, generate_output, g_pixel_values, sample_idx, ori_size, return_drafts_only=False):
        """
        Decodes LLM hidden states into binary masks via the SAM2 grounding encoder.
        """
        import os
        from PIL import Image
        ret_masks = []
        w, h = ori_size

        # 1. Extract hidden states at the positions where the LLM predicted [SEG]
        hidden_states = generate_output.hidden_states
        
        # Flattening hidden states from generation steps
        last_hidden_states = torch.cat([item[-1][0] for item in hidden_states], dim=0)

        seg_hidden_states = get_seg_hidden_states(
            last_hidden_states,
            generate_output.sequences[0][:-1],
            seg_id=self.seg_token_idx,
        )

        if seg_hidden_states.shape[0] == 0:
            debug_error("Found [SEG] tokens in sequence but get_seg_hidden_states returned empty.")
            return ret_masks

        debug_info(f"Decoding {seg_hidden_states.shape[0]} [SEG] tokens...")

        # 2. Map LLM hidden states to SAM2 embedding space
        all_seg_features = self.text_hidden_fcs(seg_hidden_states)

        for obj_idx, seg_feat in enumerate(all_seg_features):
            # Broadcast the text feature across the sampled frames
            lang_embeds = torch.cat([seg_feat.unsqueeze(0)] * len(sample_idx), dim=0)[:, None]

            # Get image features from SAM2
            sam_states = self.grounding_encoder.get_sam2_embeddings_training_like(
                g_pixel_values[sample_idx]
            )
            
            # --- STAGE 1: INITIAL DRAFT ---
            pred_masks = self.grounding_encoder.language_embd_inference(
                sam_states, lang_embeds, sample_idx
            )
            masks = F.interpolate(pred_masks, size=(h, w), mode="bilinear", align_corners=False)
            masks_np = (masks[:, 0].sigmoid() > 0.5).cpu().numpy()

            # print(len(masks))
            # print(len(masks_np))

            debug_info(f"Obj {obj_idx}: Draft mask generated. Frames: {len(masks_np)}")

            if return_drafts_only:
                ret_masks.append(masks_np)
                continue 

            # --- STAGE 2: REFINEMENT ---
            refined_pred = self.grounding_encoder.train_postprocess(g_pixel_values, masks_np, sample_idx)

            # print(len(refined_pred))

            final_masks_interp = F.interpolate(refined_pred, size=(h, w), mode="bilinear", align_corners=False)
            final_masks_np = (final_masks_interp[:, 0].sigmoid() > 0.5).cpu().numpy()
            
            # print(len(final_masks_interp))
            # print(len(final_masks_np))

            ret_masks.append(final_masks_np)

            # print(len(ret_masks))

            debug_info(f"Obj {obj_idx}: Refinement complete. Final mask shape: {final_masks_np.shape}")

        debug_info(f"Total objects decoded: {len(ret_masks)}")
        return ret_masks

    # ------------------------------------------------------------------------
    # MAIN PREDICT FORWARD
    # ------------------------------------------------------------------------

    def predict_forward(
        self,
        image=None,
        video=None,
        text=None,
        past_text="",
        mask_prompts=None,
        tokenizer=None,
        sample_num_frames=5,
        sample_idx=None,
        return_drafts_only=False,
    ):
        # 1. Initialize config if needed
        if not self.init_prediction_config:
            debug_info("Initializing prediction config...")
            self.preparing_for_generation(tokenizer=tokenizer)

        # 2. Handle input modality and determine frame sampling
        visual_input = video if video is not None else image
        
        if sample_idx is not None:
            debug_info(f"Using provided sample_idx: {sample_idx}")
        elif video is not None:
            sample_idx = np.linspace(0, len(video) - 1, sample_num_frames, dtype=int).tolist()
            debug_info(f"Sampled {sample_num_frames} frames from video")
        elif isinstance(image, list):
            sample_idx = list(range(len(image)))
            debug_info(f"Using list of {len(image)} images")
        else:
            sample_idx = [0]
            debug_info("Single image input detected")

        # 3. Process visuals
        if visual_input is not None:
            pixel_values, g_pixel_values, visual_str, vp_overall_mask, processed_masks = (
                self._process_visual_input(visual_input, sample_idx=sample_idx, mask_prompts=mask_prompts)
            )
            ori_size = visual_input[0].size if isinstance(visual_input, list) else visual_input.size
            debug_info(f"Visuals processed. Ori size: {ori_size}")
        else:
            debug_info("No visual input provided, stripping <image> tags")
            text = text.replace("<image>", "")
            pixel_values = g_pixel_values = vp_overall_mask = processed_masks = None
            visual_str = ""
            ori_size = (0, 0)

        # 4. Build text instruction
        if "<image>" in text or mask_prompts is not None:
            past_text = "" 

        processed_text = text.replace("<image>", visual_str)
        input_text = past_text + self.template["INSTRUCTION"].format(
            input=processed_text, round=1, bot_name=self.bot_name
        )

        ids = torch.tensor(self.tokenizer.encode(input_text)).cuda().unsqueeze(0)
        debug_info(f"Input IDs shape: {ids.shape}")

        # 5. Execute LLM Generation
        debug_info("Starting LLM generation...")
        generate_output = self.generate(
            input_ids=ids,
            pixel_values=pixel_values,
            attention_mask=torch.ones_like(ids, dtype=torch.bool),
            generation_config=self.gen_config,
            stopping_criteria=self.stop_criteria,
            output_hidden_states=True,
            return_dict_in_generate=True,
            prompt_masks=processed_masks,
            vp_overall_mask=vp_overall_mask,
        )

        predict = self.tokenizer.decode(generate_output.sequences[0], skip_special_tokens=False).strip()
        debug_info(f"LLM Output: {predict[:100]}...")

        # 6. Trigger SAM2 Mask Decoding
        ret_masks = []
        has_seg_token = self.seg_token_idx in generate_output.sequences[0]
        
        if pixel_values is not None and has_seg_token:
            debug_info("SEG token detected, decoding masks...")
            ret_masks = self._decode_masks(
                generate_output, 
                g_pixel_values, 
                sample_idx, 
                ori_size, 
                return_drafts_only=return_drafts_only
            )
        elif has_seg_token and pixel_values is None:
            debug_error("SEG token found but pixel_values is None. Cannot decode.")

        return {
            "prediction": predict,
            "prediction_masks": ret_masks,
            "sample_idx": sample_idx,
            "is_draft": return_drafts_only 
        }


def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    
    debug_info(f"seg_id: {seg_id} | mask_matches: {seg_mask.sum()} | n_out: {n_out}")

    if n_out == 0:
        debug_error("n_out is 0, nothing to slice")
        return hidden_states[0:0]

    if hidden_states.shape[0] < n_out:
        debug_error(f"Shape mismatch: hidden_states {hidden_states.shape[0]} < output_ids {n_out}")

    return hidden_states[-n_out:][seg_mask]