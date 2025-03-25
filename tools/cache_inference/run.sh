python tools/cache_inference/teacache_inference_consisid.py \
    --rel_l1_thresh 0.1 \
    --ckpts_path BestWishYsh/ConsisID-preview \
    --image "https://github.com/PKU-YuanGroup/ConsisID/blob/main/asserts/example_images/2.png?raw=true" \
    --prompt "The video captures a boy walking along a city street, filmed in black and white on a classic 35mm camera. His expression is thoughtful, his brow slightly furrowed as if he's lost in contemplation. The film grain adds a textured, timeless quality to the image, evoking a sense of nostalgia. Around him, the cityscape is filled with vintage buildings, cobblestone sidewalks, and softly blurred figures passing by, their outlines faint and indistinct. Streetlights cast a gentle glow, while shadows play across the boy\'s path, adding depth to the scene. The lighting highlights the boy\'s subtle smile, hinting at a fleeting moment of curiosity. The overall cinematic atmosphere, complete with classic film still aesthetics and dramatic contrasts, gives the scene an evocative and introspective feel." \
    --seed 42 \
    --num_infer_steps 50 \
    --output_path ./teacache_results