import os
from huggingface_hub import snapshot_download


if __name__ == "__main__":
    model_path = "../../ckpts"

    if not os.path.exists(os.path.join(model_path, "data_process")):
        snapshot_download(repo_id="openai/clip-vit-base-patch32", local_dir=os.path.join(model_path, "data_process", "clip-vit-base-patch32"))

    if not os.path.exists(model_path):
        snapshot_download(repo_id="BestWishYsh/ConsisID-preview", local_dir=model_path)
