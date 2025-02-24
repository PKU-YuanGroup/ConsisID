import os
import requests
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm


def download_file(url, local_path):
    if os.path.exists(local_path):
        local_file_size = os.path.getsize(local_path)
        response = requests.head(url)
        remote_file_size = int(response.headers.get('Content-Length', 0))

        if local_file_size == remote_file_size:
            print(f"{os.path.basename(local_path)} already exists with the correct size, skipping download.")
            return

    response = requests.head(url)
    file_size = int(response.headers.get('Content-Length', 0))

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192), total=file_size // 8192, unit='KB', desc=os.path.basename(local_path)):
                f.write(chunk)


if __name__ == "__main__":
    model_path = "../../ckpts"

    if not os.path.exists(os.path.join(model_path, "data_process", "step1_yolov8_face.pt")):
        hf_hub_download(repo_id="BestWishYsh/ConsisID-preview", filename="data_process/step1_yolov8_face.pt", local_dir=model_path)

    if not os.path.exists(os.path.join(model_path, "data_process", "step1_yolov8_head.pt")):
        hf_hub_download(repo_id="BestWishYsh/ConsisID-preview", filename="data_process/step1_yolov8_head.pt", local_dir=model_path)

    if not os.path.exists(os.path.join(model_path, "data_process", "sam2.1_hiera_large.pt")):
        hf_hub_download(repo_id="facebook/sam2.1-hiera-large", filename="sam2.1_hiera_large.pt", local_dir=os.path.join(model_path, "data_process"))

    if not os.path.exists(os.path.join(model_path, "data_process", "Qwen2-VL-7B-Instruct")):
        snapshot_download(repo_id="Qwen/Qwen2-VL-7B-Instruct", local_dir=os.path.join(model_path, "data_process", "Qwen2-VL-7B-Instruct"))

    if not os.path.exists(os.path.join(model_path, "data_process", "yolo11l.pt")):
        download_file("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt", os.path.join(model_path, "data_process", "yolo11l.pt"))

    if not os.path.exists(os.path.join(model_path, "data_process", "yolo11l-pose.pt")):
        download_file("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-pose.pt", os.path.join(model_path, "data_process", "yolo11l-pose.pt"))
