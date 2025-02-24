import math
import os
import random
import threading
import time
from datetime import datetime, timedelta

import gradio as gr
# import spaces
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from diffusers import ConsisIDPipeline
from diffusers.pipelines.consisid.consisid_utils import prepare_face_models, process_face_embeddings_infer
from moviepy import VideoFileClip
from util.rife_model import load_rife_model, rife_inference_with_latents
from util.utils import load_sd_upscale, save_video, upscale_batch_and_concatenate

from diffusers.image_processor import VaeImageProcessor
from diffusers.training_utils import free_memory


# 0. Pre config
model_path = "ckpts"

lora_path = None
lora_rank = 128
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists(model_path) or not os.path.exists(f"{model_path}/model_real_esran") or not os.path.exists(f"{model_path}/model_rife"):
    print("Model not found, downloading from Hugging Face...")
    hf_hub_download(repo_id="ai-forever/Real-ESRGAN", filename="RealESRGAN_x4.pth", local_dir=f"{model_path}/model_real_esran")
    snapshot_download(repo_id="AlexWortega/RIFE", local_dir=f"{model_path}/model_rife")
    snapshot_download(repo_id="BestWishYsh/ConsisID-preview", local_dir=f"{model_path}")
else:
    print(f"Model already exists in {model_path}, skipping download.")


# 1. Prepare all the face models
face_helper_1, face_helper_2, face_clip_model, face_main_model, eva_transform_mean, eva_transform_std = prepare_face_models(model_path, device, dtype)


# 2. Load Pipeline.
pipe = ConsisIDPipeline.from_pretrained(model_path, torch_dtype=dtype)

# If you're using with lora, add this code
if lora_path:
    pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
    pipe.fuse_lora(lora_scale=1 / lora_rank)


# 3. Move to device.
pipe.to(device)
# Save Memory. Turn on if you don't have multiple GPUs or enough GPU memory(such as H100) and it will cost more time in inference, it may also reduce the quality
pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
# pipe.vae.enable_slicing()
# pipe.vae.enable_tiling()

os.makedirs("./output", exist_ok=True)
os.makedirs("./gradio_tmp", exist_ok=True)

# load upscale and interpolation model
upscale_model = load_sd_upscale(f"{model_path}/model_real_esran/RealESRGAN_x4.pth", device)
frame_interpolation_model = load_rife_model(f"{model_path}/model_rife")

# @spaces.GPU(duration=180)
def generate(
    prompt: str,
    image_input: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    seed: int = 42,
    negative_prompt: str = None,
    scale_status: bool = False,
    rife_status: bool = False,
):
    if seed == -1:
        seed = random.randint(0, 2**8 - 1)

    # 4. Prepare model input
    id_cond, id_vit_hidden, image, face_kps = process_face_embeddings_infer(face_helper_1, face_clip_model, face_helper_2,
                                                                            eva_transform_mean, eva_transform_std,
                                                                            face_main_model, device, dtype,
                                                                            image_input, is_align_face=True)

    prompt = prompt.strip('"')
    if negative_prompt:
        negative_prompt = negative_prompt.strip('"')

    # 5. Generate Identity-Preserving Video
    generator = torch.Generator(device).manual_seed(seed) if seed else None
    video_pt = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        num_videos_per_prompt=1,
        num_inference_steps=num_inference_steps,
        num_frames=49,
        use_dynamic_cfg=False,
        guidance_scale=guidance_scale,
        generator=generator,
        id_vit_hidden=id_vit_hidden,
        id_cond=id_cond,
        kps_cond=face_kps,
        output_type="pt",
    ).frames

    free_memory()

    if scale_status:
        video_pt = upscale_batch_and_concatenate(upscale_model, video_pt, device)
    if rife_status:
        video_pt = rife_inference_with_latents(frame_interpolation_model, video_pt)

    batch_size = video_pt.shape[0]
    batch_video_frames = []
    for batch_idx in range(batch_size):
        pt_image = video_pt[batch_idx]
        pt_image = torch.stack([pt_image[i] for i in range(pt_image.shape[0])])

        image_np = VaeImageProcessor.pt_to_numpy(pt_image)
        image_pil = VaeImageProcessor.numpy_to_pil(image_np)
        batch_video_frames.append(image_pil)

    return (batch_video_frames, seed)


def convert_to_gif(video_path):
    clip = VideoFileClip(video_path)
    gif_path = video_path.replace(".mp4", ".gif")
    clip.write_gif(gif_path, fps=8)
    return gif_path


def delete_old_files():
    while True:
        now = datetime.now()
        cutoff = now - timedelta(minutes=10)
        directories = ["./output", "./gradio_tmp"]

        for directory in directories:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_mtime < cutoff:
                        os.remove(file_path)
        time.sleep(600)


threading.Thread(target=delete_old_files, daemon=True).start()
examples_images = [
    ["asserts/example_images/1.png", "A woman adorned with a delicate flower crown, is standing amidst a field of gently swaying wildflowers. Her eyes sparkle with a serene gaze, and a faint smile graces her lips, suggesting a moment of peaceful contentment. The shot is framed from the waist up, highlighting the gentle breeze lightly tousling her hair. The background reveals an expansive meadow under a bright blue sky, capturing the tranquility of a sunny afternoon."],
    ["asserts/example_images/2.png", "The video captures a boy walking along a city street, filmed in black and white on a classic 35mm camera. His expression is thoughtful, his brow slightly furrowed as if he's lost in contemplation. The film grain adds a textured, timeless quality to the image, evoking a sense of nostalgia. Around him, the cityscape is filled with vintage buildings, cobblestone sidewalks, and softly blurred figures passing by, their outlines faint and indistinct. Streetlights cast a gentle glow, while shadows play across the boy's path, adding depth to the scene. The lighting highlights the boy's subtle smile, hinting at a fleeting moment of curiosity. The overall cinematic atmosphere, complete with classic film still aesthetics and dramatic contrasts, gives the scene an evocative and introspective feel."],
    ["asserts/example_images/3.png", "The video depicts a man sitting at an office desk, engaged in his work. He is dressed in a formal suit and appears to be focused on his computer screen. The office environment is well-organized, with shelves filled with binders and other office supplies neatly arranged. The man is holding a red cup, possibly containing a beverage, which he drinks from before setting it down on the desk. He then proceeds to type on the keyboard, indicating that he is working on something on his computer. The overall atmosphere of the video suggests a professional setting where the man is diligently working on his tasks."],
    ["asserts/example_images/4.png", "The video features a baby wearing a bright superhero cape, standing confidently with arms raised in a powerful pose. The baby has a determined look on their face, with eyes wide and lips pursed in concentration, as if ready to take on a challenge. The setting appears playful, with colorful toys scattered around and a soft rug underfoot, while sunlight streams through a nearby window, highlighting the fluttering cape and adding to the impression of heroism. The overall atmosphere is lighthearted and fun, with the baby's expressions capturing a mix of innocence and an adorable attempt at bravery, as if truly ready to save the day."],
]

with gr.Blocks() as demo:
    gr.Markdown("""
           <div style='display: flex; align-items: center; justify-content: center; text-align: center;'>
                <img src='https://www.pnglog.com/bwrbe1.png' style='width: 150px; height: auto; margin-right: 10px;' />
           </div>
           <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
               ConsisID-Preview Space
           </div>
           <div style="text-align: center;">
               <a href="https://huggingface.co/BestWishYsh/ConsisID">🤗 Model Hub</a> |
               <a href="https://huggingface.co/datasets/BestWishYsh/ConsisID-preview-Data">📚 Dataset</a> |
               <a href="https://github.com/PKU-YuanGroup/ConsisID">🌐 Github</a> |
               <a href="https://pku-yuangroup.github.io/ConsisID">📝 Page</a> |
               <a href="https://huggingface.co/papers/2411.17440">📜 arxiv </a>
           </div>
           <div style="text-align: center;display: flex;justify-content: center;align-items: center;margin-top: 1em;margin-bottom: .5em;">
              <span>If the Space is too busy, duplicate it to use privately</span>
              <a href="https://huggingface.co/spaces/BestWishYsh/ConsisID-Space?duplicate=true"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/duplicate-this-space-lg.svg" width="160" style="
                margin-left: .75em;
            "></a>
           </div>
           <div style="text-align: center; font-size: 15px; font-weight: bold; color: red; margin-bottom: 20px;">
            ⚠️ This demo is for academic research and experiential use only. The current codes and weights are our early versions, and we will release the full codes in the next few days.
            </div>
           """)
    with gr.Row():
        with gr.Column():
            with gr.Accordion("IPT2V: Face Input", open=True):
                image_input = gr.Image(label="Input Image (should contain clear face, preferably half-body or full-body image)")
                prompt = gr.Textbox(label="Prompt (Less than 200 Words)", placeholder="Enter your prompt here. ConsisID has high requirements for prompt quality. You can use GPT-4o to refine the input text prompt, example can be found on our github.", lines=5)
                negative_prompt = gr.Textbox(label="Negative Prompt (Default is None)", placeholder="Enter your negative prompt here. Default is None", lines=1)
            with gr.Accordion("Examples", open=False):
                examples_component_images = gr.Examples(
                    examples_images,
                    inputs=[image_input, prompt],
                    cache_examples=False,
                )

            with gr.Group():
                with gr.Column():
                    num_inference_steps = gr.Slider(1, 100, value=50, step=1, label="Number of Inference Steps")
                    with gr.Row():
                        seed_param = gr.Number(
                            label="Inference Seed (Enter a positive number, -1 for random)", value=42
                        )
                        cfg_param = gr.Number(
                            label="Guidance Scale (Enter a positive number, default = 6.0)", value=6.0
                        )
                    with gr.Row():
                        enable_scale = gr.Checkbox(label="Super-Resolution (720 × 480 -> 2880 × 1920)", value=False)
                        enable_rife = gr.Checkbox(label="Frame Interpolation (8fps -> 16fps)", value=False)
                    gr.Markdown(
                        "✨In this demo, we use [RIFE](https://github.com/hzwer/ECCV2022-RIFE) for frame interpolation and [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) for upscaling(Super-Resolution)."
                    )

            generate_button = gr.Button("🎬 Generate Video")

        with gr.Column():
            video_output = gr.Video(label="ConsisID Generate Video", width=720, height=480)
            with gr.Row():
                download_video_button = gr.File(label="📥 Download Video", visible=False)
                download_gif_button = gr.File(label="📥 Download GIF", visible=False)
                seed_text = gr.Number(label="Seed Used for Video Generation", visible=False)

    gr.Markdown("""
    <table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
        <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
            🎥 Video Gallery
        </div>
        <tr>
            <td style="width: 25%; vertical-align: top; font-size: 0.9em;">
                <p>The video features a woman in exquisite hybrid armor adorned with iridescent gemstones, standing amidst gently falling cherry blossoms. Her piercing yet serene gaze hints at quiet determination, as a breeze catches a loose strand of her hair. She stands in a tranquil courtyard framed by moss-covered stone walls and wooden arches, with blossoms casting soft shadows on the ground. The petals swirl around her, adding a dreamlike quality, while the blurred backdrop emphasizes her poised figure. The scene conveys elegance, strength, and tranquil readiness, capturing a moment of peace before an upcoming challenge.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/97fa0710-4f14-4a6d-b6f7-f3a2e9f7486e" width="100%" controls autoplay loop></video>
            </td>
            <td style="width: 25%; vertical-align: top; font-size: 0.9em;">
                <p>The video features a baby wearing a bright superhero cape, standing confidently with arms raised in a powerful pose. The baby has a determined look on their face, with eyes wide and lips pursed in concentration, as if ready to take on a challenge. The setting appears playful, with colorful toys scattered around and a soft rug underfoot, while sunlight streams through a nearby window, highlighting the fluttering cape and adding to the impression of heroism. The overall atmosphere is lighthearted and fun, with the baby's expressions capturing a mix of innocence and an adorable attempt at bravery, as if truly ready to save the day.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/90b547a3-247c-4bb0-abae-ba53483b7b6e" width="100%" controls autoplay loop></video>
            </td>
        </tr>
        <tr>
            <td style="width: 25%; vertical-align: top; font-size: 0.9em;">
                <p>The video features a man standing next to an airplane, engaged in a conversation on his cell phone. he is wearing sunglasses and a black top, and he appears to be talking seriously. The airplane has a green stripe running along its side, and there is a large engine visible behind his. The man seems to be standing near the entrance of the airplane, possibly preparing to board or just having disembarked. The setting suggests that he might be at an airport or a private airfield. The overall atmosphere of the video is professional and focused, with the man's attire and the presence of the airplane indicating a business or travel context.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/55680c58-de86-48b4-8d86-e9906a3185c3" width="100%" controls autoplay loop></video>
            </td>
            <td style="width: 25%; vertical-align: top; font-size: 0.9em;">
                <p>The video features a woman with blonde hair standing on a beach near the water's edge. She is wearing a black swimsuit and appears to be enjoying her time by the sea. The sky above is clear with some clouds, and the ocean waves gently lap against the shore. The woman seems to be holding something white in her hand, possibly a piece of driftwood or a small object found on the beach. The overall atmosphere of the video is serene and relaxing, capturing the beauty of nature and the simple pleasure of being by the ocean.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/8d06e702-f80e-4cb2-abc2-b0f519ec3f11" width="100%" controls autoplay loop></video>
            </td>
        </tr>
        <tr>
            <td style="width: 25%; vertical-align: top; font-size: 0.9em;">
                <p>The video features a man sitting in a red armchair, enjoying a cup of coffee or tea. he is dressed in a light-colored outfit and has long dark-haired hair. The setting appears to be indoors, with large windows providing a view of a misty or foggy coastal landscape outside. The room has a modern design with geometric structures visible in the background. There is a small round table next to the armchair, also holding a cup. The overall atmosphere suggests a calm and serene moment, possibly during a cold or rainy day by the sea.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/ab9c655e-84c2-47ed-85d9-039a7f64adfe" width="100%" controls autoplay loop></video>
            </td>
            <td style="width: 25%; vertical-align: top; font-size: 0.9em;">
                <p>The video shows a young boy sitting at a table, eating a piece of food. He appears to be enjoying his meal, as he takes a bite and chews it. The boy is wearing a blue shirt and has short hair. The background is dark, with some light coming from the left side of the frame. There is a straw visible on the right side of the frame, suggesting that there may be a drink next to the boy's plate. The overall atmosphere of the video seems casual and relaxed, with the focus on the boy's enjoyment of his food.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/8014b02e-e1c4-4df7-b7f3-cebfb01fa373" width="100%" controls autoplay loop></video>
            </td>
        </tr>
        <tr>
            <td style="width: 25%; vertical-align: top; font-size: 0.9em;">
                <p>The video captures a boy walking along a city street, filmed in black and white on a classic 35mm camera. His expression is thoughtful, his brow slightly furrowed as if he's lost in contemplation. The film grain adds a textured, timeless quality to the image, evoking a sense of nostalgia. Around him, the cityscape is filled with vintage buildings, cobblestone sidewalks, and softly blurred figures passing by, their outlines faint and indistinct. Streetlights cast a gentle glow, while shadows play across the boy's path, adding depth to the scene. The lighting highlights the boy's subtle smile, hinting at a fleeting moment of curiosity. The overall cinematic atmosphere, complete with classic film still aesthetics and dramatic contrasts, gives the scene an evocative and introspective feel.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/e4bc3169-d3d4-46e2-a667-8b456ead9465" width="100%" controls autoplay loop></video>
            </td>
            <td style="width: 25%; vertical-align: top; font-size: 0.9em;">
                <p>The video features a young man standing outdoors in a snowy park. he is wearing a colorful winter jacket with a floral pattern and a white knit hat. The background shows a snowy landscape with trees, benches, and a metal fence. The ground is covered in snow, and there is a light snowfall in the air. The man appears to be enjoying the winter weather, as he smiles and gives a thumbs-up gesture towards the camera. The overall atmosphere of the video is cheerful and festive, capturing the beauty of a snowy day in a park.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/e4e3e519-95d4-44e0-afa7-9a833f99e090" width="100%" controls autoplay loop></video>
            </td>
        </tr>
    </table>
        """)


    def run(
        prompt,
        negative_prompt,
        image_input,
        num_inference_steps,
        seed_value,
        cfg_param,
        scale_status,
        rife_status,
        progress=gr.Progress(track_tqdm=True)
    ):
        batch_video_frames, seed = generate(
            prompt,
            image_input,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=cfg_param,
            seed=seed_value,
            scale_status=scale_status,
            rife_status=rife_status,
        )

        video_path = save_video(batch_video_frames[0], fps=math.ceil((len(batch_video_frames[0]) - 1) / 6))
        video_update = gr.update(visible=True, value=video_path)
        gif_path = convert_to_gif(video_path)
        gif_update = gr.update(visible=True, value=gif_path)
        seed_update = gr.update(visible=True, value=seed)

        return video_path, video_update, gif_update, seed_update

    generate_button.click(
        fn=run,
        inputs=[prompt, negative_prompt, image_input, num_inference_steps, seed_param, cfg_param, enable_scale, enable_rife],
        outputs=[video_output, download_video_button, download_gif_button, seed_text],
    )


if __name__ == "__main__":
    demo.queue(max_size=15)
    demo.launch()