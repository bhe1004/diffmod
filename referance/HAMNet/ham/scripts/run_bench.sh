#!/bin/bash

# List of test domains
domains=("duktig" "basket" "suitcase" "circular_bin" "drawer" "flat" "grill" "high_cab" "tight_cab")
num_steps=2050
# Iterate through each domain
for domain in "${domains[@]}"; do
    python3 test.py +test="$domain" ++env.seed=56381 \
    ++global_device=cuda:0 \
    ++env.num_env=346 \
    ++load_ckpt=HAMNet/public:pretrained \
    ++test_steps=$num_steps ++draw_debug_lines=0 ++env.use_viewer=0 \
    ++env.single_object_scene.load_episode="/tmp/docker/bench_eps/${domain}-qrand-128.pth" \
    ++log_categorical_results=1 ++cat_result_output="/tmp/docker/result/ours/${domain}"
    
done
