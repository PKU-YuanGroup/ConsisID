# <u>Evaluation Pipeline</u> by *ConsisID*
This repo describes how to evaluate customized model in the [ConsisID](https://arxiv.org/abs/2411.17440) paper.

## ⚙️ Download Data and Weight

The Evaluate Data is available at [HuggingFace](https://huggingface.co/datasets/BestWishYsh/ConsisID-preview-Data), which will be used to sample videos by your own models. The weights will be automatically downloaded, or you can download it with the following commands.

```bash
cd util
python download_weights_eval.py
```

Once ready, the weights will be organized in this format:

```
📦 ConsisiID/
├── 📂 ckpts/
│   ├── 📂 data_process/
│       ├── 📂 clip-vit-base-patch32
│   ├── 📂 face_encoder/
```

## 🗝️ Usage

### Way 1 - Step by step

```
# Get FaceSim-Score and FID-Score
python get_facesim_fid.py
# Get CLIPScore
python get_clipscore.py
```

### Way 2 - All in once

```
pip install natsort
python eval_all_in_once.py
```

We would like to thank [@z-jiaming](https://github.com/z-jiaming) for his work on this [eval_all-in-once.py](https://github.com/PKU-YuanGroup/ConsisID/tree/main/eval/eval_all-in-once.py).

## 🔒 Limitation

- The currently released [video_caption_eval_old.csv](https://huggingface.co/datasets/BestWishYsh/ConsisID-preview-Data/blob/main/video_caption_eval_old.csv) is of low quality, and we further performed prompt refine on it in the article.  And we will release the latest csv in the future.
- The current code has not yet standardized the output format and currently only supports measuring a single video or prompt. We will continue to update the code in the future.
