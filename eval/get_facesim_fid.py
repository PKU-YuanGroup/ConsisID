import os
import sys

import cv2
import numpy
import numpy as np
import torch
from huggingface_hub import snapshot_download
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from PIL import Image
from torchvision import models, transforms


current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(os.path.dirname(current_file_path))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from util.on_going_module.curricularface import get_model


def load_image(image):
    img = image.convert('RGB')
    img = transforms.Resize((299, 299))(img)  # Resize to Inception input size
    img = transforms.ToTensor()(img)
    return img.unsqueeze(0)  # Add batch dimension


def get_activations(images, model, batch_size=16):
    model.eval()
    activations = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            pred = model(batch)
            activations.append(pred)
    activations = torch.cat(activations, dim=0).cpu().numpy()
    if activations.shape[0] == 1:
        activations = np.repeat(activations, 2, axis=0)
    return activations


def matrix_sqrt(matrix):
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    sqrt_eigenvalues = torch.sqrt(torch.clamp(eigenvalues, min=0))
    sqrt_matrix = (eigenvectors * sqrt_eigenvalues).mm(eigenvectors.T)
    return sqrt_matrix


def calculate_fid(real_activations, fake_activations, device="cuda"):
    real_activations_tensor = torch.tensor(real_activations).to(device)
    fake_activations_tensor = torch.tensor(fake_activations).to(device)

    mu1 = real_activations_tensor.mean(dim=0)
    sigma1 = torch.cov(real_activations_tensor.T)
    mu2 = fake_activations_tensor.mean(dim=0)
    sigma2 = torch.cov(fake_activations_tensor.T)

    ssdiff = torch.sum((mu1 - mu2) ** 2)
    covmean = matrix_sqrt(sigma1.mm(sigma2))
    if torch.is_complex(covmean):
        covmean = covmean.real
    fid = ssdiff + torch.trace(sigma1 + sigma2 - 2 * covmean)
    return fid.item()


@torch.no_grad()
def inference(face_model, img, device):
    img = cv2.resize(img, (112, 112))
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float().to(device)
    img.div_(255).sub_(0.5).div_(0.5)
    embedding = face_model(img).detach().cpu().numpy()[0]
    return embedding / np.linalg.norm(embedding)


def pad_np_bgr_image(np_image, scale=1.25):
    assert scale >= 1.0, "scale should be >= 1.0"
    pad_scale = scale - 1.0
    h, w = np_image.shape[:2]
    top = bottom = int(h * pad_scale)
    left = right = int(w * pad_scale)
    return cv2.copyMakeBorder(np_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128)), (left, top)


def get_face_keypoints(face_model, image_bgr):
    face_info = face_model.get(image_bgr)
    if len(face_info) > 0:
        return sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
    return None


def process_image(face_model, image_path):
    if isinstance(image_path, str):
        np_faceid_image = np.array(Image.open(image_path).convert("RGB"))
    elif isinstance(image_path, numpy.ndarray):
        np_faceid_image = image_path
    else:
        raise TypeError("image_path should be a string or PIL.Image.Image object")

    image_bgr = cv2.cvtColor(np_faceid_image, cv2.COLOR_RGB2BGR)

    face_info = get_face_keypoints(face_model, image_bgr)
    if face_info is None:
        padded_image, sub_coord = pad_np_bgr_image(image_bgr)
        face_info = get_face_keypoints(face_model, padded_image)
        if face_info is None:
            print("Warning: No face detected in the image. Continuing processing...")
            return None, None
        face_kps = face_info['kps']
        face_kps -= np.array(sub_coord)
    else:
        face_kps = face_info['kps']
    arcface_embedding = face_info['embedding']

    norm_face = face_align.norm_crop(image_bgr, landmark=face_kps, image_size=224)
    align_face = cv2.cvtColor(norm_face, cv2.COLOR_BGR2RGB)

    return align_face, arcface_embedding


def sample_video_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def batch_cosine_similarity(embedding_image, embedding_frames, device="cuda"):
    embedding_image = torch.tensor(embedding_image).to(device)
    embedding_frames = torch.tensor(embedding_frames).to(device)
    return torch.nn.functional.cosine_similarity(embedding_image, embedding_frames, dim=-1).cpu().numpy()


# def process_video(video_path, cur_image_embedding, face_arc_model, face_cur_model, device, arcface_image_embedding, real_activations, fid_model):
def process_video(video_path, face_arc_model, face_cur_model, fid_model, arcface_image_embedding, cur_image_embedding, real_activations, device):
    video_frames = sample_video_frames(video_path, num_frames=16)

    # Initialize lists to store the scores
    cur_scores = []
    arc_scores = []
    fid_face = []

    for frame in video_frames:
        # Convert to RGB once at the beginning
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for ArcFace embeddings
        align_face_frame, arcface_frame_embedding = process_image(face_arc_model, frame_rgb)

        # Skip if alignment fails
        if align_face_frame is None:
            continue

        # Perform inference for current face model
        cur_embedding_frame = inference(face_cur_model, align_face_frame, device)

        # Compute cosine similarity for cur_score and arc_score in a compact manner
        cur_score = max(0.0, batch_cosine_similarity(cur_image_embedding, cur_embedding_frame, device=device).item())
        arc_score = max(0.0, batch_cosine_similarity(arcface_image_embedding, arcface_frame_embedding, device=device).item())

        # Process FID score
        align_face_frame_pil = Image.fromarray(align_face_frame)
        fake_image = load_image(align_face_frame_pil).to(device)
        fake_activations = get_activations(fake_image, fid_model)
        fid_score = calculate_fid(real_activations, fake_activations, device)

        # Collect scores
        fid_face.append(fid_score)
        cur_scores.append(cur_score)
        arc_scores.append(arc_score)

    # Aggregate results with default values for empty lists
    avg_cur_score = np.mean(cur_scores) if cur_scores else 0.0
    avg_arc_score = np.mean(arc_scores) if arc_scores else 0.0
    avg_fid_score = np.mean(fid_face) if fid_face else 0.0

    return avg_cur_score, avg_arc_score, avg_fid_score


def get_facesim_fid(
    device = "cuda",
    model_path = "../ckpts",
    video_path = "path/your.mp4",
    image_path = "path/your.png",
    results_file_path = "facesim_fid_score.txt",
):

    if not os.path.exists(model_path):
        print("Model not found, downloading from Hugging Face...")
        snapshot_download(repo_id="BestWishYsh/ConsisID-preview", local_dir=model_path)
    else:
        print(f"Model already exists in {model_path}, skipping download.")

    face_arc_path = os.path.join(model_path, "face_encoder")
    face_cur_path = os.path.join(face_arc_path, "glint360k_curricular_face_r101_backbone.bin")

    # Initialize FaceEncoder model for face detection and embedding extraction
    face_arc_model = FaceAnalysis(root=face_arc_path, providers=['CUDAExecutionProvider'])
    face_arc_model.prepare(ctx_id=0, det_size=(320, 320))

    # Load face recognition model
    face_cur_model = get_model('IR_101')([112, 112])
    face_cur_model.load_state_dict(torch.load(face_cur_path, map_location="cpu"))
    face_cur_model = face_cur_model.to(device)
    face_cur_model.eval()

    # Load InceptionV3 model for FID calculation
    fid_model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    fid_model.fc = torch.nn.Identity()  # Remove final classification layer
    fid_model.eval()
    fid_model = fid_model.to(device)

    # Process the single video and image pair
    # Extract embeddings and features from the image
    align_face_image, arcface_image_embedding = process_image(face_arc_model, image_path)
    if align_face_image is None:
        print(f"Error processing image at {image_path}")
        return

    cur_image_embedding = inference(face_cur_model, align_face_image, device)
    align_face_image_pil = Image.fromarray(align_face_image)
    real_image = load_image(align_face_image_pil).to(device)
    real_activations = get_activations(real_image, fid_model)

    # Process the video and calculate scores
    cur_score, arc_score, fid_score = process_video(
        video_path, face_arc_model, face_cur_model, fid_model,
        arcface_image_embedding, cur_image_embedding, real_activations, device
    )

    # Write results to file
    with open(results_file_path, 'w') as f:
        f.write(f"cur score: {cur_score}\n")
        f.write(f"arc score: {arc_score}\n")
        f.write(f"fid score: {fid_score}\n")

    # Print results
    print(f"cur score: {cur_score}")
    print(f"arc score: {arc_score}")
    print(f"fid score: {fid_score}")

    return cur_score, arc_score, fid_score


if __name__ == "__main__":
    get_facesim_fid()
