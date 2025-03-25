import os

import cv2
from huggingface_hub import snapshot_download
from transformers import CLIPModel, CLIPProcessor


def compute_clip_score(video_path, model, processor, prompt, device, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []

    for i in range(num_frames):
        frame_idx = int(i * total_frames / num_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            print(f"Warning: Frame at index {frame_idx} could not be read.")

    cap.release()

    if model is not None:
        inputs = processor(text=prompt, images=frames, return_tensors="pt", padding=True, truncation=True)
        inputs.to(device)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        average_score = logits_per_image.mean().item()

    return average_score


def get_clipscore(
    device = "cuda",
    model_path = "../ckpts/data_process/clip-vit-base-patch32",
    prompt = "your prompt",
    video_file_path = "path/your.mp4",
    results_file_path = "clipscore.txt",
):

    if not os.path.exists(model_path):
        print("Model not found, downloading from Hugging Face...")
        snapshot_download(repo_id="openai/clip-vit-base-patch32", local_dir=model_path)
    else:
        print(f"Model already exists in {model_path}, skipping download.")

    clip_model = CLIPModel.from_pretrained(model_path)
    clip_processor = CLIPProcessor.from_pretrained(model_path)
    clip_model.to(device)

    clip_score = compute_clip_score(video_file_path, clip_model, clip_processor, prompt, device, num_frames=16)

    # Write results to file
    with open(results_file_path, 'w') as f:
        f.write(f"clip score: {clip_score}\n")

    # Print results
    print(f"clip score: {clip_score}")
    return clip_score


if __name__ == "__main__":
    get_clipscore()
