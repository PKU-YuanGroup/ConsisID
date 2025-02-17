import os
from typing import List, Optional, Tuple, Union

import cv2
import insightface
import numpy as np
import torch
from consisid_eva_clip import create_model_and_transforms
from consisid_eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from insightface.app import FaceAnalysis
from PIL import Image, ImageOps
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize
from transformers import T5EncoderModel, T5Tokenizer

from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.utils import load_image


###### pipeline ###
def resize_numpy_image_long(image, resize_long_edge=768):
    """
    Resize the input image to a specified long edge while maintaining aspect ratio.

    Args:
        image (numpy.ndarray): Input image (H x W x C or H x W).
        resize_long_edge (int): The target size for the long edge of the image. Default is 768.

    Returns:
        numpy.ndarray: Resized image with the long edge matching `resize_long_edge`, while maintaining the aspect
        ratio.
    """

    h, w = image.shape[:2]
    if max(h, w) <= resize_long_edge:
        return image
    k = resize_long_edge / max(h, w)
    h = int(h * k)
    w = int(w * k)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == "float64":
                img = img.astype("float32")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    return _totensor(imgs, bgr2rgb, float32)


def to_gray(img):
    """
    Converts an RGB image to grayscale by applying the standard luminosity formula.

    Args:
        img (torch.Tensor): The input image tensor with shape (batch_size, channels, height, width).
                             The image is expected to be in RGB format (3 channels).

    Returns:
        torch.Tensor: The grayscale image tensor with shape (batch_size, 3, height, width).
                      The grayscale values are replicated across all three channels.
    """
    x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    x = x.repeat(1, 3, 1, 1)
    return x


def process_face_embeddings(
    face_helper_1,
    clip_vision_model,
    face_helper_2,
    eva_transform_mean,
    eva_transform_std,
    app,
    device,
    weight_dtype,
    image,
    original_id_image=None,
    is_align_face=True,
):
    """
    Process face embeddings from an image, extracting relevant features such as face embeddings, landmarks, and parsed
    face features using a series of face detection and alignment tools.

    Args:
        face_helper_1: Face helper object (first helper) for alignment and landmark detection.
        clip_vision_model: Pre-trained CLIP vision model used for feature extraction.
        face_helper_2: Face helper object (second helper) for embedding extraction.
        eva_transform_mean: Mean values for image normalization before passing to EVA model.
        eva_transform_std: Standard deviation values for image normalization before passing to EVA model.
        app: Application instance used for face detection.
        device: Device (CPU or GPU) where the computations will be performed.
        weight_dtype: Data type of the weights for precision (e.g., `torch.float32`).
        image: Input image in RGB format with pixel values in the range [0, 255].
        original_id_image: (Optional) Original image for feature extraction if `is_align_face` is False.
        is_align_face: Boolean flag indicating whether face alignment should be performed.

    Returns:
        Tuple:
            - id_cond: Concatenated tensor of Ante face embedding and CLIP vision embedding
            - id_vit_hidden: Hidden state of the CLIP vision model, a list of tensors.
            - return_face_features_image_2: Processed face features image after normalization and parsing.
            - face_kps: Keypoints of the face detected in the image.
    """

    face_helper_1.clean_all()
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # get antelopev2 embedding
    face_info = app.get(image_bgr)
    if len(face_info) > 0:
        face_info = sorted(face_info, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))[
            -1
        ]  # only use the maximum face
        id_ante_embedding = face_info["embedding"]  # (512,)
        face_kps = face_info["kps"]
    else:
        id_ante_embedding = None
        face_kps = None

    # using facexlib to detect and align face
    face_helper_1.read_image(image_bgr)
    face_helper_1.get_face_landmarks_5(only_center_face=True)
    if face_kps is None:
        face_kps = face_helper_1.all_landmarks_5[0]
    face_helper_1.align_warp_face()
    if len(face_helper_1.cropped_faces) == 0:
        raise RuntimeError("facexlib align face fail")
    align_face = face_helper_1.cropped_faces[0]  # (512, 512, 3)  # RGB

    # incase insightface didn't detect face
    if id_ante_embedding is None:
        print("fail to detect face using insightface, extract embedding on align face")
        id_ante_embedding = face_helper_2.get_feat(align_face)

    id_ante_embedding = torch.from_numpy(id_ante_embedding).to(device, weight_dtype)  # torch.Size([512])
    if id_ante_embedding.ndim == 1:
        id_ante_embedding = id_ante_embedding.unsqueeze(0)  # torch.Size([1, 512])

    # parsing
    if is_align_face:
        input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0  # torch.Size([1, 3, 512, 512])
        input = input.to(device)
        parsing_out = face_helper_1.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
        parsing_out = parsing_out.argmax(dim=1, keepdim=True)  # torch.Size([1, 1, 512, 512])
        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        bg = sum(parsing_out == i for i in bg_label).bool()
        white_image = torch.ones_like(input)  # torch.Size([1, 3, 512, 512])
        # only keep the face features
        return_face_features_image = torch.where(bg, white_image, to_gray(input))  # torch.Size([1, 3, 512, 512])
        return_face_features_image_2 = torch.where(bg, white_image, input)  # torch.Size([1, 3, 512, 512])
    else:
        original_image_bgr = cv2.cvtColor(original_id_image, cv2.COLOR_RGB2BGR)
        input = img2tensor(original_image_bgr, bgr2rgb=True).unsqueeze(0) / 255.0  # torch.Size([1, 3, 512, 512])
        input = input.to(device)
        return_face_features_image = return_face_features_image_2 = input

    # transform img before sending to eva-clip-vit
    face_features_image = resize(
        return_face_features_image, clip_vision_model.image_size, InterpolationMode.BICUBIC
    )  # torch.Size([1, 3, 336, 336])
    face_features_image = normalize(face_features_image, eva_transform_mean, eva_transform_std)
    id_cond_vit, id_vit_hidden = clip_vision_model(
        face_features_image.to(weight_dtype), return_all_features=False, return_hidden=True, shuffle=False
    )  # torch.Size([1, 768]),  list(torch.Size([1, 577, 1024]))
    id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, True)
    id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)

    id_cond = torch.cat(
        [id_ante_embedding, id_cond_vit], dim=-1
    )  # torch.Size([1, 512]), torch.Size([1, 768])  ->  torch.Size([1, 1280])

    return (
        id_cond,
        id_vit_hidden,
        return_face_features_image_2,
        face_kps,
    )  # torch.Size([1, 1280]), list(torch.Size([1, 577, 1024]))


def process_face_embeddings_infer(
    face_helper_1,
    clip_vision_model,
    face_helper_2,
    eva_transform_mean,
    eva_transform_std,
    app,
    device,
    weight_dtype,
    img_file_path,
    is_align_face=True,
):
    """
    Process face embeddings from an input image for inference, including alignment, feature extraction, and embedding
    concatenation.

    Args:
        face_helper_1: Face helper object (first helper) for alignment and landmark detection.
        clip_vision_model: Pre-trained CLIP vision model used for feature extraction.
        face_helper_2: Face helper object (second helper) for embedding extraction.
        eva_transform_mean: Mean values for image normalization before passing to EVA model.
        eva_transform_std: Standard deviation values for image normalization before passing to EVA model.
        app: Application instance used for face detection.
        device: Device (CPU or GPU) where the computations will be performed.
        weight_dtype: Data type of the weights for precision (e.g., `torch.float32`).
        img_file_path: Path to the input image file (string) or a numpy array representing an image.
        is_align_face: Boolean flag indicating whether face alignment should be performed (default: True).

    Returns:
        Tuple:
            - id_cond: Concatenated tensor of Ante face embedding and CLIP vision embedding.
            - id_vit_hidden: Hidden state of the CLIP vision model, a list of tensors.
            - image: Processed face image after feature extraction and alignment.
            - face_kps: Keypoints of the face detected in the image.
    """

    # Load and preprocess the input image
    if isinstance(img_file_path, str):
        image = np.array(load_image(image=img_file_path).convert("RGB"))
    else:
        image = np.array(ImageOps.exif_transpose(Image.fromarray(img_file_path)).convert("RGB"))

    # Resize image to ensure the longer side is 1024 pixels
    image = resize_numpy_image_long(image, 1024)
    original_id_image = image

    # Process the image to extract face embeddings and related features
    id_cond, id_vit_hidden, align_crop_face_image, face_kps = process_face_embeddings(
        face_helper_1,
        clip_vision_model,
        face_helper_2,
        eva_transform_mean,
        eva_transform_std,
        app,
        device,
        weight_dtype,
        image,
        original_id_image,
        is_align_face,
    )

    # Convert the aligned cropped face image (torch tensor) to a numpy array
    tensor = align_crop_face_image.cpu().detach()
    tensor = tensor.squeeze()
    tensor = tensor.permute(1, 2, 0)
    tensor = tensor.numpy() * 255
    tensor = tensor.astype(np.uint8)
    image = ImageOps.exif_transpose(Image.fromarray(tensor))

    return id_cond, id_vit_hidden, image, face_kps


def prepare_face_models(model_path, device, dtype):
    """
    Prepare all face models for the facial recognition task.

    Parameters:
    - model_path: Path to the directory containing model files.
    - device: The device (e.g., 'cuda', 'cpu') where models will be loaded.
    - dtype: Data type (e.g., torch.float32) for model inference.

    Returns:
    - face_helper_1: First face restoration helper.
    - face_helper_2: Second face restoration helper.
    - face_clip_model: CLIP model for face extraction.
    - eva_transform_mean: Mean value for image normalization.
    - eva_transform_std: Standard deviation value for image normalization.
    - face_main_model: Main face analysis model.
    """
    # get helper model
    face_helper_1 = FaceRestoreHelper(
        upscale_factor=1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model="retinaface_resnet50",
        save_ext="png",
        device=device,
        model_rootpath=os.path.join(model_path, "face_encoder"),
    )
    face_helper_1.face_parse = None
    face_helper_1.face_parse = init_parsing_model(
        model_name="bisenet", device=device, model_rootpath=os.path.join(model_path, "face_encoder")
    )
    face_helper_2 = insightface.model_zoo.get_model(
        f"{model_path}/face_encoder/models/antelopev2/glintr100.onnx", providers=["CUDAExecutionProvider"]
    )
    face_helper_2.prepare(ctx_id=0)

    # get local facial extractor part 1
    model, _, _ = create_model_and_transforms(
        "EVA02-CLIP-L-14-336",
        os.path.join(model_path, "face_encoder", "EVA02_CLIP_L_336_psz14_s6B.pt"),
        force_custom_clip=True,
    )
    face_clip_model = model.visual
    eva_transform_mean = getattr(face_clip_model, "image_mean", OPENAI_DATASET_MEAN)
    eva_transform_std = getattr(face_clip_model, "image_std", OPENAI_DATASET_STD)
    if not isinstance(eva_transform_mean, (list, tuple)):
        eva_transform_mean = (eva_transform_mean,) * 3
    if not isinstance(eva_transform_std, (list, tuple)):
        eva_transform_std = (eva_transform_std,) * 3
    eva_transform_mean = eva_transform_mean
    eva_transform_std = eva_transform_std

    # get local facial extractor part 2
    face_main_model = FaceAnalysis(
        name="antelopev2", root=os.path.join(model_path, "face_encoder"), providers=["CUDAExecutionProvider"]
    )
    face_main_model.prepare(ctx_id=0, det_size=(640, 640))

    # move face models to device
    face_helper_1.face_det.eval()
    face_helper_1.face_parse.eval()
    face_clip_model.eval()
    face_helper_1.face_det.to(device)
    face_helper_1.face_parse.to(device)
    face_clip_model.to(device, dtype=dtype)

    return face_helper_1, face_helper_2, face_clip_model, face_main_model, eva_transform_mean, eva_transform_std



###### train ###
def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    """
    Generate prompt embeddings using the T5 model for a given prompt or list of prompts.

    Args:
        tokenizer (T5Tokenizer): Tokenizer used to encode the text prompt(s).
        text_encoder (T5EncoderModel): Pretrained T5 encoder model to generate embeddings.
        prompt (Union[str, List[str]]): Single prompt or list of prompts to encode.
        num_videos_per_prompt (int, optional): Number of video embeddings to generate per prompt. Defaults to 1.
        max_sequence_length (int, optional): Maximum length for the tokenized prompt. Defaults to 226.
        device (Optional[torch.device], optional): The device on which to run the model (e.g., "cuda", "cpu").
        dtype (Optional[torch.dtype], optional): The data type for the embeddings (e.g., torch.float32).
        text_input_ids (optional): Pre-tokenized input IDs. If not provided, tokenizer is used to encode the prompt.

    Returns:
        torch.Tensor: The generated prompt embeddings reshaped for the specified number of video generations per prompt.
    """

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


def encode_prompt(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    """
    Encode the given prompt(s) into embeddings using the T5 model.

    This function wraps the _get_t5_prompt_embeds function to generate prompt embeddings
    for a given prompt or list of prompts. It allows for generating multiple embeddings
    per prompt, useful for tasks like video generation.

    Args:
        tokenizer (T5Tokenizer): Tokenizer used to encode the text prompt(s).
        text_encoder (T5EncoderModel): Pretrained T5 encoder model to generate embeddings.
        prompt (Union[str, List[str]]): Single prompt or list of prompts to encode.
        num_videos_per_prompt (int, optional): Number of video embeddings to generate per prompt. Defaults to 1.
        max_sequence_length (int, optional): Maximum length for the tokenized prompt. Defaults to 226.
        device (Optional[torch.device], optional): The device on which to run the model (e.g., "cuda", "cpu").
        dtype (Optional[torch.dtype], optional): The data type for the embeddings (e.g., torch.float32).
        text_input_ids (optional): Pre-tokenized input IDs. If not provided, tokenizer is used to encode the prompt.

    Returns:
        torch.Tensor: The generated prompt embeddings reshaped for the specified number of video generations per prompt.
    """

    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds = _get_t5_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
        text_input_ids=text_input_ids,
    )
    return prompt_embeds


def compute_prompt_embeddings(
    tokenizer, text_encoder, prompt, max_sequence_length, device, dtype, requires_grad: bool = False
):
    """
    Compute the prompt embeddings based on whether gradient computation is required.

    This function generates embeddings for a given prompt or list of prompts, either
    with or without gradient tracking, depending on the `requires_grad` argument. It
    uses the `encode_prompt` function to generate embeddings for the provided prompt(s).

    Args:
        tokenizer (T5Tokenizer): Tokenizer used to encode the text prompt(s).
        text_encoder (T5EncoderModel): Pretrained T5 encoder model to generate embeddings.
        prompt (Union[str, List[str]]): Single prompt or list of prompts to encode.
        max_sequence_length (int): Maximum length for the tokenized prompt.
        device (torch.device): The device on which to run the model (e.g., "cuda", "cpu").
        dtype (torch.dtype): The data type for the embeddings (e.g., torch.float32).
        requires_grad (bool, optional): Whether the embeddings should require gradient computation. Defaults to False.

    Returns:
        torch.Tensor: The generated prompt embeddings.
    """

    if requires_grad:
        prompt_embeds = encode_prompt(
            tokenizer,
            text_encoder,
            prompt,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
    else:
        with torch.no_grad():
            prompt_embeds = encode_prompt(
                tokenizer,
                text_encoder,
                prompt,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
    return prompt_embeds


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    patch_size_t: int = None,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    if patch_size_t is None:
        # Version 1.0
        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
        )
    else:
        # Version 1.5
        base_num_frames = (num_frames + patch_size_t - 1) // patch_size_t

        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(base_size_height, base_size_width),
        )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin


def tensor_to_pil(src_img_tensor):
    """
    Converts a tensor image to a PIL image.

    This function takes an input tensor with the shape (C, H, W) and converts it
    into a PIL Image format. It ensures that the tensor is in the correct data
    type and moves it to CPU if necessary.

    Parameters:
        src_img_tensor (torch.Tensor): Input image tensor with shape (C, H, W),
            where C is the number of channels, H is the height, and W is the width.

    Returns:
        PIL.Image: The converted image in PIL format.
    """

    img = src_img_tensor.clone().detach()
    if img.dtype == torch.bfloat16:
        img = img.to(torch.float32)
    img = img.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.uint8)
    pil_image = Image.fromarray(img)
    return pil_image
