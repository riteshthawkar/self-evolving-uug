import math
from copy import deepcopy
from io import BytesIO
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

import numpy as np
import torch
from transformers.image_utils import get_image_size, to_numpy_array
from typing_extensions import override

from ..extras.constants import IGNORE_INDEX, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER, IMAGE_GEN_PLACEHOLDER
from ..extras.packages import is_pillow_available, is_pyav_available, is_transformers_version_greater_than
from icecream import ic
from torchvision.transforms import InterpolationMode, transforms

if is_pillow_available():
    from PIL import Image
    from PIL.Image import Image as ImageObject


if is_pyav_available():
    import av


if is_transformers_version_greater_than("4.45.0"):
    from transformers.models.mllama.processing_mllama import (
        convert_sparse_cross_attention_mask_to_dense,
        get_cross_attention_token_mask,
    )


if TYPE_CHECKING:
    from av.stream import Stream
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.image_processing_utils import BaseImageProcessor

    class EncodedImage(TypedDict):
        path: Optional[str]
        bytes: Optional[bytes]

    ImageInput = Union[str, bytes, EncodedImage, ImageObject]
    VideoInput = str


def _get_paligemma_token_type_ids(
    imglens: Sequence[int], seqlens: Sequence[int], processor: "ProcessorMixin"
) -> List[List[int]]:
    r"""
    Gets paligemma token type ids for computing loss.

    Returns:
        batch_token_type_ids: shape (batch_size, sequence_length)
    """
    batch_token_type_ids = []
    for imglen, seqlen in zip(imglens, seqlens):
        image_seqlen = imglen * getattr(processor, "image_seqlen")
        batch_token_type_ids.append([0] * image_seqlen + [1] * (seqlen - image_seqlen))

    return batch_token_type_ids


class BasePlugin:
    def __init__(self, image_token: Optional[str], video_token: Optional[str]) -> None:
        self.image_token = image_token
        self.video_token = video_token

    def _validate_input(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
    ) -> None:
        r"""
        Validates if this model accepts the input modalities.
        """
        if len(images) != 0 and self.image_token is None:
            raise ValueError("This model does not support image input.")

        if len(videos) != 0 and self.video_token is None:
            raise ValueError("This model does not support video input.")

    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        r"""
        Pre-processes a single image.
        """
        image_resolution: int = kwargs.get("image_resolution")
        if (image.width * image.height) > image_resolution:
            resize_factor = math.sqrt(image_resolution / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            from PIL import ImageFile
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def _get_video_sample_frames(self, video_stream: "Stream", **kwargs) -> int:
        r"""
        Computes video sample frames according to fps.
        """
        video_fps: float = kwargs.get("video_fps")
        video_maxlen: int = kwargs.get("video_maxlen")
        total_frames = video_stream.frames
        sample_frames = float(video_stream.duration * video_stream.time_base) * video_fps
        sample_frames = min(total_frames, video_maxlen, sample_frames)
        return math.floor(sample_frames)

    def _regularize_images(self, images: Sequence["ImageInput"], **kwargs) -> List["ImageObject"]:
        r"""
        Regularizes images to avoid error. Including reading and pre-processing.
        """
        results = []
        for image in images:
            if isinstance(image, str):
                image = Image.open(image)
            elif isinstance(image, bytes):
                image = Image.open(BytesIO(image))
            elif isinstance(image, dict):
                if image["bytes"] is not None:
                    image = Image.open(BytesIO(image["bytes"]))
                else:
                    image = Image.open(image["path"])

            if not isinstance(image, ImageObject) and image is not None:
                raise ValueError(f"Expect input is a list of Images, but got {type(image)}.")
            if image is not None:
                results.append(self._preprocess_image(image, **kwargs))

        return results

    def _regularize_videos(self, videos: Sequence["VideoInput"], **kwargs) -> List[List["ImageObject"]]:
        r"""
        Regularizes videos to avoid error. Including reading, resizing and converting.
        """
        results = []
        for video in videos:
            container = av.open(video, "r")
            video_stream = next(stream for stream in container.streams if stream.type == "video")
            total_frames = video_stream.frames
            sample_frames = self._get_video_sample_frames(video_stream, **kwargs)
            sample_indices = np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)
            frames: List["ImageObject"] = []
            container.seek(0)
            for frame_idx, frame in enumerate(container.decode(video_stream)):
                if frame_idx in sample_indices:
                    frames.append(frame.to_image())

            frames = self._regularize_images(frames, **kwargs)
            results.append(frames)

        return results

    def _get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: "ProcessorMixin",
    ) -> Dict[str, "torch.Tensor"]:
        r"""
        Processes visual inputs.

        Returns: (llava and paligemma)
            pixel_values: tensor with shape (B, C, H, W)

        Returns: (qwen2-vl)
            pixel_values: tensor with shape (num_patches, patch_dim)
            image_grid_thw: tensor with shape (num_images, 3), where the three numbers are time, width, height

        It holds num_patches == torch.prod(image_grid_thw)
        """
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        video_processor: "BaseImageProcessor" = getattr(processor, "video_processor", image_processor)
        input_dict = {"images": None}  # default key
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_resolution=getattr(processor, "image_resolution", 512 * 512),
            )
            input_dict["images"] = images

        if len(videos) != 0:
            videos = self._regularize_videos(
                videos,
                image_resolution=getattr(processor, "video_resolution", 128 * 128),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 64),
            )
            input_dict["videos"] = videos

        mm_inputs = {}
        if image_processor != video_processor:
            if input_dict.get("images") is not None:
                mm_inputs.update(image_processor(input_dict["images"], return_tensors="pt"))
            if input_dict.get("videos") is not None:
                mm_inputs.update(video_processor(input_dict["videos"], return_tensors="pt"))
        elif input_dict.get("images") is not None or input_dict.get("videos") is not None:  # same processor (qwen2-vl)
            if len(input_dict.get("images"))!=0:
                mm_inputs.update(image_processor(**input_dict, return_tensors="pt"))

        return mm_inputs

    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        r"""
        Pre-processes input messages before tokenization for VLMs.
        """
        self._validate_input(images, videos)
        return messages

    def process_token_ids(
        self,
        input_ids: List[int],
        labels: Optional[List[int]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
    ) -> Tuple[List[int], Optional[List[int]]]:
        r"""
        Pre-processes token ids after tokenization for VLMs.
        """
        self._validate_input(images, videos)
        return input_ids, labels

    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        batch_ids: Sequence[List[int]],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        r"""
        Builds batched multimodal inputs for VLMs.

        Arguments:
            images: a list of image inputs, shape (num_images,)
            videos: a list of video inputs, shape (num_videos,)
            imglens: number of images in each sample, shape (batch_size,)
            vidlens: number of videos in each sample, shape (batch_size,)
            batch_ids: token ids of input samples, shape (batch_size, seq_len)
            processor: a processor for pre-processing images and videos
        """
        self._validate_input(images, videos)
        return {}


class LlavaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        num_image_tokens = 0
        image_seqlen = getattr(processor, "image_seqlen")
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            # ic(content)
            while IMAGE_PLACEHOLDER in content:
                num_image_tokens += 1
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)
            message["content"] = content.replace("{{image}}", self.image_token)
        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens.")

        return messages

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        batch_ids: Sequence[List[int]],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        return self._get_mm_inputs(images, videos, processor)





class VargptLlavaPlugin(BasePlugin):
    @override
    def __init__(self, image_token: Optional[str], video_token: Optional[str], image_gen_token: Optional[str], image_gen_token_num: int=680) -> None:
        super().__init__(image_token, video_token)
        self.image_gen_token = image_gen_token
        self.image_gen_token_num = image_gen_token_num
        self.train_aug, self.val_aug = self.build_transforms()
    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        images = [x for x in images if x is not None]

        num_image_tokens = 0
        image_seqlen = 576
        messages = deepcopy(messages)
        image_gen_index = self.find_placeholder_positions(''.join([message["content"] for message in messages]))
        for message in messages:
            content = message["content"]
            # ic(content)
            while IMAGE_PLACEHOLDER in content:
                num_image_tokens += 1
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)
            message["content"] = content.replace("{{image}}", self.image_token)
            content = message["content"]
            while IMAGE_GEN_PLACEHOLDER in content:
                content = content.replace(
                    IMAGE_GEN_PLACEHOLDER,
                    "<|image_gen_start|>{}<|image_gen_end|>".format(
                        self.image_gen_token * self.image_gen_token_num
                    ),
                    1,
                )
            message["content"] = content
        if len(images) != num_image_tokens + len(image_gen_index):
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens.")

        return messages
    def split_images_by_sequence(self, images, sequence):
        if len(sequence)==0: 
            return [], images
        images_0 = [img for i, img in enumerate(images) if sequence[i] == 0]
        images_1 = [img for i, img in enumerate(images) if sequence[i] == 1]
        return images_0, images_1
    def find_placeholder_positions(self, text):
        normal_placeholder = IMAGE_PLACEHOLDER
        gen_placeholder = IMAGE_GEN_PLACEHOLDER
        
        all_positions = []
        gen_positions = []
        
        pos = 0
        while True:
            pos = text.find(normal_placeholder, pos)
            if pos == -1:
                break
            all_positions.append(pos)
            pos += 1
        
        pos = 0
        while True:
            pos = text.find(gen_placeholder, pos)
            if pos == -1:
                break
            all_positions.append(pos)
            gen_positions.append(pos)
            pos += 1
        
        all_positions.sort()
        
        result = []
        for gen_pos in gen_positions:
            result.append(all_positions.index(gen_pos))
        
        return result
    def build_transforms(
        self, final_reso=256,
        hflip=False, mid_reso=1.125,
    ):
        # build augmentations
        mid_reso = round(mid_reso * final_reso)
        train_aug, val_aug = [
            transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
            transforms.RandomCrop((final_reso, final_reso)),
            transforms.ToTensor(), self.normalize_01_into_pm1,
        ], [
            transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
            transforms.CenterCrop((final_reso, final_reso)),
            transforms.ToTensor(), self.normalize_01_into_pm1,
        ]
        if hflip: train_aug.insert(0, transforms.RandomHorizontalFlip())
        train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)
        return train_aug, val_aug
    def normalize_01_into_pm1(self, x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
        return x.add(x).add_(-1)
    def image_transform(self, image):
        return self.train_aug(image)
    def find_token_sequence(self, batch_ids):
        result = []
        target_tokens = [32000, 32002] #  "image_token_index": 32000,  "image_gen_start_token_id": 32002
        
        for sequence in batch_ids:
            found_positions = []
            for i, token in enumerate(sequence):
                if token in target_tokens:
                    found_positions.append((i, 0 if token == 32000 else 1))
            
            result.extend([x[1] for x in sorted(found_positions, key=lambda x: x[0])])
        
        return result
    def find_token_sequence_substr(self, batch_ids):
        result = []
        token_counts = []  
        target_tokens = [32000, 32002]
        
        for sequence in batch_ids:
            i = 0
            while i < len(sequence):
                if sequence[i] in target_tokens:
                    current_token = sequence[i]
                    consecutive_count = 1
                    while i + consecutive_count < len(sequence) and sequence[i + consecutive_count] == current_token:
                        consecutive_count += 1
                    
                    result.append(0 if current_token == 32000 else 1)
                    token_counts.append(consecutive_count) 
                    i += consecutive_count
                else:
                    i += 1
        
        return result, token_counts
    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        batch_ids: Sequence[List[int]],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        return self._get_mm_inputs_gen(images, videos, processor, batch_ids)


    def _get_mm_inputs_gen(
        self,
        images: Sequence["ImageInput"], 
        videos: Sequence["VideoInput"],
        processor: "ProcessorMixin",
        batch_ids: Sequence[List[int]] = None, # list [B, Seq]
    ) -> Dict[str, "torch.Tensor"]:
        r"""
        Processes visual inputs.

        Returns: (llava and paligemma)
            pixel_values: tensor with shape (B, C, H, W)

        Returns: (qwen2-vl)
            pixel_values: tensor with shape (num_patches, patch_dim)
            image_grid_thw: tensor with shape (num_images, 3), where the three numbers are time, width, height

        It holds num_patches == torch.prod(image_grid_thw)
        """
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        video_processor: "BaseImageProcessor" = getattr(processor, "video_processor", image_processor)
        token_sequence,_ = self.find_token_sequence_substr(batch_ids)
        images = [x for x in images if x is not None]
        assert len(images) >= len(token_sequence), (
            f"Expected {len(images)} images, but got {len(token_sequence)} images."
        )
        image_understand, image_gen = self.split_images_by_sequence(images, token_sequence)
        
        input_dict = {"images": None}  # default key
        if len(image_understand) != 0:
            image_understand = self._regularize_images(
                image_understand,
                image_resolution=getattr(processor, "image_resolution", 512 * 512),
            )
            input_dict["images"] = image_understand

        if len(videos) != 0:
            videos = self._regularize_videos(
                videos,
                image_resolution=getattr(processor, "video_resolution", 128 * 128),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 64),
            )
            input_dict["videos"] = videos

        mm_inputs = {}
        if image_processor != video_processor:
            if input_dict.get("images") is not None:
                mm_inputs.update(image_processor(input_dict["images"], return_tensors="pt"))
            if input_dict.get("videos") is not None:
                mm_inputs.update(video_processor(input_dict["videos"], return_tensors="pt"))
        elif input_dict.get("images") is not None or input_dict.get("videos") is not None:  # same processor (qwen2-vl)
            mm_inputs.update(image_processor(**input_dict, return_tensors="pt"))
        # TODO
        if len(image_gen) !=0:
            image_gen = self._regularize_images(
                image_gen,
                image_resolution=getattr(processor, "image_resolution", 512 * 512),
            )
            if self.image_transform is not None:
                image_gen = [self.image_transform(img) for img in image_gen]
                input_dict["images_gen"] = image_gen

        if len(image_gen)!=0:
            mm_inputs.update({"pixel_gen_values": torch.stack(input_dict["images_gen"], dim=0)})
        return mm_inputs

class LlavaNextPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        num_image_tokens = 0
        messages = deepcopy(messages)
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        if "image_sizes" in mm_inputs:
            image_sizes = iter(mm_inputs["image_sizes"])

        if "pixel_values" in mm_inputs:
            height, width = get_image_size(to_numpy_array(mm_inputs["pixel_values"][0][0]))

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_size = next(image_sizes)
                orig_height, orig_width = image_size
                image_seqlen = processor._get_number_of_features(orig_height, orig_width, height, width)
                if getattr(processor, "vision_feature_select_strategy") == "default":
                    image_seqlen -= 1

                num_image_tokens += 1
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)

            message["content"] = content.replace("{{image}}", self.image_token)

        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens.")

        return messages

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        batch_ids: Sequence[List[int]],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        return self._get_mm_inputs(images, videos, processor)


class LlavaNextVideoPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        if "pixel_values" in mm_inputs:
            image_sizes = iter(mm_inputs["image_sizes"])
            height, width = get_image_size(to_numpy_array(mm_inputs["pixel_values"][0][0]))
            for message in messages:
                content = message["content"]
                while IMAGE_PLACEHOLDER in content:
                    image_size = next(image_sizes)
                    orig_height, orig_width = image_size
                    image_seqlen = processor._get_number_of_features(orig_height, orig_width, height, width)
                    if getattr(processor, "vision_feature_select_strategy") == "default":
                        image_seqlen -= 1

                    num_image_tokens += 1
                    content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)

                message["content"] = content.replace("{{image}}", self.image_token)

        if "pixel_values_videos" in mm_inputs:
            pixel_values_video = to_numpy_array(mm_inputs.get("pixel_values_videos")[0])
            height, width = get_image_size(pixel_values_video[0])
            num_frames = pixel_values_video.shape[0]  # frame dim is always after batch dim
            image_seqlen = (height // processor.patch_size) * (width // processor.patch_size)
            video_seqlen = image_seqlen // 4 * num_frames  # divide by 4 needed for avg pooling layer
            for message in messages:
                content = message["content"]
                while VIDEO_PLACEHOLDER in content:
                    num_video_tokens += 1
                    content = content.replace(VIDEO_PLACEHOLDER, "{{video}}" * video_seqlen, 1)

                message["content"] = content.replace("{{video}}", self.video_token)

        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens.")

        if len(videos) != num_video_tokens:
            raise ValueError(f"The number of videos does not match the number of {VIDEO_PLACEHOLDER} tokens.")

        return messages

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        batch_ids: Sequence[List[int]],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        return self._get_mm_inputs(images, videos, processor)


class PaliGemmaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        num_image_tokens = 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                num_image_tokens += 1
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}", 1)

            message["content"] = content.replace("{{image}}", "")

        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens.")

        return messages

    @override
    def process_token_ids(
        self,
        input_ids: List[int],
        labels: Optional[List[int]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
    ) -> Tuple[List[int], Optional[List[int]]]:
        self._validate_input(images, videos)
        num_images = len(images)
        image_seqlen = num_images * getattr(processor, "image_seqlen")
        image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        input_ids = [image_token_id] * image_seqlen + input_ids
        if labels is not None:
            labels = [IGNORE_INDEX] * image_seqlen + labels

        return input_ids, labels

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        batch_ids: Sequence[List[int]],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        seqlens = [len(input_ids) for input_ids in batch_ids]
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        mm_inputs["token_type_ids"] = _get_paligemma_token_type_ids(imglens, seqlens, processor)
        return mm_inputs


class PixtralPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        patch_size = getattr(processor, "patch_size")
        image_token = getattr(processor, "image_token")
        image_break_token = getattr(processor, "image_break_token")
        image_end_token = getattr(processor, "image_end_token")

        num_image_tokens = 0
        messages = deepcopy(messages)
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        image_input_sizes = mm_inputs.get("image_sizes", None)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if image_input_sizes is None:
                    raise ValueError("Cannot get image input sizes.")

                image_size = image_input_sizes[0][num_image_tokens]
                height, width = image_size
                num_height_tokens = height // patch_size
                num_width_tokens = width // patch_size
                replace_tokens = [[image_token] * num_width_tokens + [image_break_token]] * num_height_tokens
                replace_tokens = [item for sublist in replace_tokens for item in sublist]  # flatten list
                replace_tokens[-1] = image_end_token
                replace_str = "".join(replace_tokens)
                content = content.replace(IMAGE_PLACEHOLDER, replace_str, 1)
                num_image_tokens += 1

            message["content"] = content

        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens.")

        return messages

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        batch_ids: Sequence[List[int]],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        if mm_inputs.get("pixel_values"):
            mm_inputs["pixel_values"] = mm_inputs["pixel_values"][0]

        mm_inputs.pop("image_sizes", None)
        return mm_inputs


class Qwen2vlPlugin(BasePlugin):
    @override
    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        image = super()._preprocess_image(image, **kwargs)
        if min(image.width, image.height) < 28:
            width, height = max(image.width, 28), max(image.height, 28)
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.width / image.height > 200:
            width, height = image.height * 180, image.height
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.height / image.width > 200:
            width, height = image.width, image.width * 180
            image = image.resize((width, height), resample=Image.NEAREST)

        return image

    @override
    def _get_video_sample_frames(self, video_stream: "Stream", **kwargs) -> int:
        sample_frames = super()._get_video_sample_frames(video_stream, **kwargs)
        sample_frames = sample_frames // 2 * 2
        return sample_frames

    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        merge_length: int = getattr(image_processor, "merge_size") ** 2
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        image_grid_thw = mm_inputs.get("image_grid_thw", [])
        video_grid_thw = mm_inputs.get("video_grid_thw", [])

        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if num_image_tokens >= len(image_grid_thw):
                    raise ValueError(f"`len(images)` is less than the number of {IMAGE_PLACEHOLDER} tokens.")

                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    "<|vision_start|>{}<|vision_end|>".format(
                        self.image_token * (image_grid_thw[num_image_tokens].prod() // merge_length)
                    ),
                    1,
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                if num_video_tokens >= len(video_grid_thw):
                    raise ValueError(f"`len(videos)` is less than the number of {VIDEO_PLACEHOLDER} tokens.")

                content = content.replace(
                    VIDEO_PLACEHOLDER,
                    "<|vision_start|>{}<|vision_end|>".format(
                        self.video_token * (video_grid_thw[num_video_tokens].prod() // merge_length)
                    ),
                    1,
                )
                num_video_tokens += 1

            message["content"] = content

        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens.")

        if len(videos) != num_video_tokens:
            raise ValueError(f"The number of videos does not match the number of {VIDEO_PLACEHOLDER} tokens.")

        return messages

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        batch_ids: Sequence[List[int]],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        return self._get_mm_inputs(images, videos, processor)


class VideoLlavaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        num_frames = 0
        has_images = "pixel_values_images" in mm_inputs
        has_videos = "pixel_values_videos" in mm_inputs
        if has_images or has_videos:
            if has_images:
                height, width = get_image_size(to_numpy_array(mm_inputs.get("pixel_values_images")[0]))
                num_frames = 1

            if has_videos:
                pixel_values_video = to_numpy_array(mm_inputs.get("pixel_values_videos")[0])
                height, width = get_image_size(pixel_values_video[0])
                num_frames = pixel_values_video.shape[0]  # frame dim is always after batch dim

            image_seqlen = (height // processor.patch_size) * (width // processor.patch_size) + 1
            video_seqlen = image_seqlen * num_frames
            if getattr(processor, "vision_feature_select_strategy") == "default":
                image_seqlen -= 1

            for message in messages:
                content = message["content"]
                while IMAGE_PLACEHOLDER in content:
                    num_image_tokens += 1
                    content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)

                while VIDEO_PLACEHOLDER in content:
                    num_video_tokens += 1
                    content = content.replace(VIDEO_PLACEHOLDER, "{{video}}" * video_seqlen, 1)

                content = content.replace("{{image}}", self.image_token)
                message["content"] = content.replace("{{video}}", self.video_token)

        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens.")

        if len(videos) != num_video_tokens:
            raise ValueError(f"The number of videos does not match the number of {VIDEO_PLACEHOLDER} tokens.")

        return messages

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        batch_ids: Sequence[List[int]],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        return self._get_mm_inputs(images, videos, processor)


class MllamaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        num_image_tokens = 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            num_image_tokens += content.count(IMAGE_PLACEHOLDER)
            message["content"] = content.replace(IMAGE_PLACEHOLDER, self.image_token)

        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens.")

        return messages

    @override
    def _get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: "ProcessorMixin",
    ) -> Dict[str, "torch.Tensor"]:
        r"""
        Processes visual inputs for mllama because its image processor only accepts List[List[ImageInput]].

        Returns:
            pixel_values: tensor with shape
                          (batch_size, max_num_images, max_image_tiles, channels, tile_height, tile_width)
                          For example, (2, 1, 4, 3, 560, 560).
            aspect_ratio_ids: tensor with shape (batch_size, max_num_images). For example, (2, 1).
            aspect_ratio_mask: tensor with shape (batch_size, max_num_images, max_image_tiles). For example, (2, 1, 4).
            num_tiles: List[List[int]] with shape (batch_size, num_images_in_batch). For example, (2, 1).
        """
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        images = self._regularize_images(images, image_resolution=getattr(processor, "image_resolution", 512 * 512))
        return image_processor([[image] for image in images], return_tensors="pt")

    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        batch_ids: Sequence[List[int]],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        if len(images) != len(batch_ids):
            raise ValueError("Mllama only supports one image per sample.")

        mm_inputs = self._get_mm_inputs(images, videos, processor)
        num_tiles = mm_inputs.pop("num_tiles")
        image_token_id = getattr(processor, "image_token_id")
        max_image_tiles = getattr(processor.image_processor, "max_image_tiles")
        cross_attention_token_mask = [
            get_cross_attention_token_mask(input_ids, image_token_id) for input_ids in batch_ids
        ]
        mm_inputs["cross_attention_mask"] = torch.from_numpy(
            convert_sparse_cross_attention_mask_to_dense(
                cross_attention_token_mask,
                num_tiles=num_tiles,
                max_num_tiles=max_image_tiles,
                length=max(len(input_ids) for input_ids in batch_ids),
            )
        )  # shape: (batch_size, length, max_num_images, max_num_tiles)
        return mm_inputs

class VargptQwen2vlPlugin(BasePlugin):
    @override
    def __init__(self, image_token: Optional[str], video_token: Optional[str], image_gen_token: Optional[str], image_gen_token_num: int=680) -> None:
        super().__init__(image_token, video_token)
        self.image_gen_token = image_gen_token
        self.image_gen_token_num = image_gen_token_num
        self.image_gen_resolution = 512 # 256 512
        self.train_aug, self.val_aug = self.build_transforms(final_reso=self.image_gen_resolution)
    @override
    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        image = super()._preprocess_image(image, **kwargs)
        if min(image.width, image.height) < 28:
            width, height = max(image.width, 28), max(image.height, 28)
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.width / image.height > 200:
            width, height = image.height * 180, image.height
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.height / image.width > 200:
            width, height = image.width, image.width * 180
            image = image.resize((width, height), resample=Image.NEAREST)

        return image

    @override
    def _get_video_sample_frames(self, video_stream: "Stream", **kwargs) -> int:
        sample_frames = super()._get_video_sample_frames(video_stream, **kwargs)
        sample_frames = sample_frames // 2 * 2
        return sample_frames

    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        # images = [x for x in images if x is not None]

        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        merge_length: int = getattr(image_processor, "merge_size") ** 2
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        image_grid_thw = mm_inputs.get("image_grid_thw", [])
        video_grid_thw = mm_inputs.get("video_grid_thw", [])

        images = [x for x in images if x is not None]

        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)

        image_gen_index = self.find_placeholder_positions(''.join([message["content"] for message in messages]))
        if len(image_grid_thw)>0:
            image_grid_thw = image_grid_thw[torch.ones(len(image_grid_thw), dtype=torch.bool).index_fill_(0, torch.tensor(image_gen_index, dtype=torch.int64), False)]
        if len(video_grid_thw)>0:
            video_grid_thw = video_grid_thw[torch.ones(len(video_grid_thw), dtype=torch.bool).index_fill_(0, torch.tensor(image_gen_index, dtype=torch.int64), False)]
        
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if num_image_tokens >= len(image_grid_thw):
                    raise ValueError(f"`len(images)` is less than the number of {IMAGE_PLACEHOLDER} tokens.")

                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    "<|vision_start|>{}<|vision_end|>".format(
                        self.image_token * (image_grid_thw[num_image_tokens].prod() // merge_length)
                    ),
                    1,
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                if num_video_tokens >= len(video_grid_thw):
                    raise ValueError(f"`len(videos)` is less than the number of {VIDEO_PLACEHOLDER} tokens.")

                content = content.replace(
                    VIDEO_PLACEHOLDER,
                    "<|vision_start|>{}<|vision_end|>".format(
                        self.video_token * (video_grid_thw[num_video_tokens].prod() // merge_length)
                    ),
                    1,
                )
                num_video_tokens += 1
            
            while IMAGE_GEN_PLACEHOLDER in content:
                content = content.replace(
                    IMAGE_GEN_PLACEHOLDER,
                    "<|image_gen_start|>{}<|image_gen_end|>".format(
                        self.image_gen_token * self.image_gen_token_num
                    ),
                    1,
                )

            message["content"] = content
        # print(f"{images}, num_image_tokens:{num_image_tokens}, num_video_tokens:{num_video_tokens}, image_gen_index:{image_gen_index}, len(image_gen_index):{len(image_gen_index)} ")
        if len(images) != num_image_tokens + len(image_gen_index):
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens.")

        if len(videos) != num_video_tokens :
            raise ValueError(f"The number of videos does not match the number of {VIDEO_PLACEHOLDER} tokens.")

        return messages
    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        batch_ids: Sequence[List[int]],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        return self._get_mm_inputs_gen(images, videos, processor, batch_ids)

    def _get_mm_inputs_gen(
        self,
        images: Sequence["ImageInput"], 
        videos: Sequence["VideoInput"],
        processor: "ProcessorMixin",
        batch_ids: Sequence[List[int]] = None, # list [B, Seq]
    ) -> Dict[str, "torch.Tensor"]:
        r"""
        Processes visual inputs.

        Returns: (llava and paligemma)
            pixel_values: tensor with shape (B, C, H, W)

        Returns: (qwen2-vl)
            pixel_values: tensor with shape (num_patches, patch_dim)
            image_grid_thw: tensor with shape (num_images, 3), where the three numbers are time, width, height

        It holds num_patches == torch.prod(image_grid_thw)
        """
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        video_processor: "BaseImageProcessor" = getattr(processor, "video_processor", image_processor)
        token_sequence = self.find_token_sequence(batch_ids)
        images = [x for x in images if x is not None]
        image_understand, image_gen = self.split_images_by_sequence(images, token_sequence)
        input_dict = {"images": None}  # default key
        if len(image_understand) != 0:
            image_understand = self._regularize_images(
                image_understand,
                image_resolution=getattr(processor, "image_resolution", 512 * 512),
            )
            input_dict["images"] = image_understand

        if len(videos) != 0:
            videos = self._regularize_videos(
                videos,
                image_resolution=getattr(processor, "video_resolution", 128 * 128),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 64),
            )
            input_dict["videos"] = videos

        mm_inputs = {}
        if image_processor != video_processor:
            if input_dict.get("images") is not None:
                mm_inputs.update(image_processor(input_dict["images"], return_tensors="pt"))
            if input_dict.get("videos") is not None:
                mm_inputs.update(video_processor(input_dict["videos"], return_tensors="pt"))
        elif input_dict.get("images") is not None or input_dict.get("videos") is not None:  # same processor (qwen2-vl)
            mm_inputs.update(image_processor(**input_dict, return_tensors="pt"))

        # TODO
        if len(image_gen) !=0:
            image_gen = self._regularize_images(
                image_gen,
                image_resolution=getattr(processor, "image_resolution", 512 * 512),
            )
            if self.image_transform is not None:
                image_gen = [self.image_transform(img) for img in image_gen]
                input_dict["images_gen"] = image_gen

        if len(image_gen)!=0:
            mm_inputs.update({"pixel_gen_values": torch.stack(input_dict["images_gen"], dim=0)})
        return mm_inputs


    def find_token_sequence(self, batch_ids):
        result = []
        target_tokens = [151652, 151657]
        
        for sequence in batch_ids:
            found_positions = []
            for i, token in enumerate(sequence):
                if token in target_tokens:
                    found_positions.append((i, 0 if token == 151652 else 1))
            
            result.extend([x[1] for x in sorted(found_positions, key=lambda x: x[0])])
        
        return result
    
    def normalize_01_into_pm1(self, x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
        return x.add(x).add_(-1)
    def build_transforms(
        self, final_reso=256,
        hflip=False, mid_reso=1.125,
    ):
        # build augmentations
        mid_reso = round(mid_reso * final_reso)
        train_aug, val_aug = [
            transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
            transforms.RandomCrop((final_reso, final_reso)),
            transforms.ToTensor(), self.normalize_01_into_pm1,
        ], [
            transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
            transforms.CenterCrop((final_reso, final_reso)),
            transforms.ToTensor(), self.normalize_01_into_pm1,
        ]
        if hflip: train_aug.insert(0, transforms.RandomHorizontalFlip())
        train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)
        return train_aug, val_aug
    
    def image_transform(self, image):
        return self.train_aug(image)
    def split_images_by_sequence(self, images, sequence):
        if len(sequence)==0:
            return [], images
        if len(sequence)> len(images):
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens.")
        elif len(sequence)< len(images):
            print(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens.")
            clone_images = images[:len(sequence)]
        else:
            clone_images = images
        images_0 = [img for i, img in enumerate(clone_images) if sequence[i] == 0]
        images_1 = [img for i, img in enumerate(clone_images) if sequence[i] == 1]
        return images_0, images_1

    def find_placeholder_positions(self, text):
        normal_placeholder = IMAGE_PLACEHOLDER
        gen_placeholder = IMAGE_GEN_PLACEHOLDER
        
        all_positions = []
        gen_positions = []
        
        pos = 0
        while True:
            pos = text.find(normal_placeholder, pos)
            if pos == -1:
                break
            all_positions.append(pos)
            pos += 1
        
        pos = 0
        while True:
            pos = text.find(gen_placeholder, pos)
            if pos == -1:
                break
            all_positions.append(pos)
            gen_positions.append(pos)
            pos += 1
        
        all_positions.sort()
        
        result = []
        for gen_pos in gen_positions:
            result.append(all_positions.index(gen_pos))
        
        return result



PLUGINS = {
    "base": BasePlugin,
    "llava": LlavaPlugin,
    "llava_next": LlavaNextPlugin,
    "llava_next_video": LlavaNextVideoPlugin,
    "paligemma": PaliGemmaPlugin,
    "pixtral": PixtralPlugin,
    "qwen2_vl": Qwen2vlPlugin,
    "video_llava": VideoLlavaPlugin,
    "mllama": MllamaPlugin,
    "vargpt_qwen2_vl": VargptQwen2vlPlugin,
    "vargpt_llava": VargptLlavaPlugin,
}


def get_mm_plugin(
    name: str,
    image_token: Optional[str] = None,
    video_token: Optional[str] = None,
    image_gen_token: Optional[str] = None,
    image_gen_token_num: Optional[int] = None,
) -> "BasePlugin":
    plugin_class = PLUGINS.get(name, None)
    if plugin_class is None:
        raise ValueError(f"Multimodal plugin `{name}` not found.")

    if name in ["vargpt_qwen2_vl"]:
        return plugin_class(image_token, video_token, image_gen_token, image_gen_token_num)
    elif name in ["vargpt_llava"]:
        return plugin_class(image_token, video_token, image_gen_token, image_gen_token_num)

    return plugin_class(image_token, video_token)
