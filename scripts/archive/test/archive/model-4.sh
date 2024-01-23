#! /bin/bash

# simplest completionformer with RGB input

python main.py --dir_data /root/autodl-tmp/yiming/datasets/polar_hammer \
                --data_name HAMMER \
                --data_txt /root/autodl-tmp/yiming/ikemura_ws/PDNE/data_paths/hammer_MODE.txt \
                --gpus 4,5 \
                --loss 1.0*L1+1.0*L2 \
                --log_dir ./experiments/ \
                --save model-4-1-0 \
                --model VPT-V2 \
                --completionformer_mode rgbd \
                --pre_pvt \
                --pre_res \
                --pretrained_completionformer /root/autodl-tmp/yiming/PDNE/pretrained/comp/NYUv2.pt \
                --use_pol \
                --pol_rep grayscale-4 \
                --test_only \
                --data_percentage 0.05 \
                --pretrain_list_file /root/autodl-tmp/yiming/ikemura_ws/PDNE_CONCAT_eval/PDNE/scripts/test/ckpt_list/model-4.txt \
                # --pretrain /root/autodl-tmp/yiming/ikemura_ws/PDNE_VPT/PDNE/experiments/230919_040259_model-2/model_00020.pt \
                