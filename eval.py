import argparse
import os

import torch
from huggingface_hub import snapshot_download

import pandas as pd
from natsort import natsorted
import statistics
from infer import generate_video
from eval.get_facesim_fid import get_facesim_fid
from eval.get_clipscore import get_clipscore


def main(args):
    # Open video_caption_eval_old.csv
    df = pd.read_csv(os.path.join(args.eval_data_path, "video_caption_eval_old.csv"))

    img_dir = os.path.join(args.eval_data_path, "eval/face_images")

    (
        output_path_dict,
        cur_score_dict,
        arc_score_dict,
        fid_score_dict,
        clip_score_dict,
    ) = ([], [], [], [], [])

    for img_name in natsorted(os.listdir(img_dir)):
        img_file_path = os.path.join(img_dir, img_name)
        img_idx = int(img_name.split("_")[0])

        for idx in range(df.shape[0]):
            # index,Class,Original_Prompt,Refined_man,Refined_woman
            df_index = df.loc[idx, "index"]
            df_Class = df.loc[idx, "Class"]
            # df_Original_Prompt = df.loc[idx, "Original_Prompt"]
            df_Refined_man = df.loc[idx, "Refined_man"].replace(
                "{class_token}", df_Class
            )
            df_Refined_woman = df.loc[idx, "Refined_woman"].replace(
                "{class_token}", df_Class
            )

            prompt = df_Refined_woman if "woman" in img_name else df_Refined_man
            output_path = os.path.join(args.output_path, f"{img_name}--{df_index}")

            print(
                f"Processing -- {img_name} [{img_idx - 1}/{len(os.listdir(img_dir))}] -- {idx}/{df.shape[0]} in {output_path}"
            )
            print(f"Prompt: {prompt}")

            # Generate video
            output_video_path = generate_video(
                prompt=prompt,
                # negative_prompt=args.negative_prompt,
                model_path=args.model_path,
                lora_path=args.lora_path,
                lora_rank=args.lora_rank,
                output_path=output_path,  # output_path=args.output_path,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                num_videos_per_prompt=args.num_videos_per_prompt,
                dtype=torch.float16 if args.dtype == "float16" else torch.bfloat16,
                seed=args.seed,
                img_file_path=img_file_path,
            )

            # Get FaceSim-Score and FID-Score
            cur_score, arc_score, fid_score = get_facesim_fid(
                device="cuda",
                model_path=args.model_path,
                video_path=output_video_path,
                image_path=img_file_path,
                results_file_path=f"{output_video_path}_facesim_fid_score.txt",
            )
            # Get CLIPScore
            clip_score = get_clipscore(
                device="cuda",
                model_path=f"{args.model_path}/data_process/clip-vit-base-patch32",
                prompt=prompt,
                video_file_path=output_video_path,
                results_file_path=f"{output_video_path}_clipscore.txt",
            )

            output_path_dict.append(output_path)
            cur_score_dict.append(cur_score)
            arc_score_dict.append(arc_score)
            fid_score_dict.append(fid_score)
            clip_score_dict.append(clip_score)

    # Average scores
    output_path_dict.append("Average")
    cur_score_dict.append(statistics.mean(cur_score_dict))
    arc_score_dict.append(statistics.mean(arc_score_dict))
    fid_score_dict.append(statistics.mean(fid_score_dict))
    clip_score_dict.append(statistics.mean(clip_score_dict))

    # Save scores to csv
    df_output = pd.DataFrame(
        {
            "output_path": output_path_dict,
            "cur_score": cur_score_dict,
            "arc_score": arc_score_dict,
            "fid_score": fid_score_dict,
            "clip_score": clip_score_dict,
        }
    )
    df_output.to_csv(os.path.join(args.output_path, "output_scores.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Eval your custom model."
    )

    # ckpt arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default="./ckpts",
        help="The path of the pre-trained model to be used",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="The path of the LoRA weights to be used",
    )
    parser.add_argument(
        "--lora_rank", type=int, default=128, help="The rank of the LoRA weights"
    )
    # input arguments
    parser.add_argument("--eval_data_path", type=str, default="ConsisID-preview-Data")
    # output arguments
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output",
        help="The path where the generated video will be saved",
    )
    # generation arguments
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=6.0,
        help="The scale for classifier-free guidance",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of steps for the inference process",
    )
    parser.add_argument(
        "--num_videos_per_prompt",
        type=int,
        default=1,
        help="Number of videos to generate per prompt",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="The data type for computation (e.g., 'float16' or 'bfloat16')",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="The seed for reproducibility"
    )

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print("Base Model not found, downloading from Hugging Face...")
        snapshot_download(
            repo_id="BestWishYsh/ConsisID-preview", local_dir=args.model_path
        )
    else:
        print(f"Base Model already exists in {args.model_path}, skipping download.")

    main(args)
