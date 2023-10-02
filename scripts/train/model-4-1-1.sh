#! /bin/bash

# simplest completionformer with RGB input

python main.py --dir_data /root/autodl-tmp/yiming/datasets/polar_hammer \
                --data_name HAMMER \
                --data_txt /root/autodl-tmp/yiming/ikemura_ws/PDNE/data_paths/hammer_MODE.txt \
                --gpus 7,9 \
                --loss 1.0*L1+1.0*L2 \
                --batch_size 14 \
                --epochs 250 \
                --log_dir ./experiments/ \
                --save model-4-1-1 \
                --model VPT-V2 \
                --completionformer_mode rgbd \
                --pre_pvt \
                --pre_res \
                --save_full \
                --warm_up \
                --lr 0.000805 \
                --pretrained_completionformer /root/autodl-tmp/yiming/PDNE/pretrained/comp/NYUv2.pt \
                --use_pol \
                --pol_rep grayscale-4 \
                # --resume --pretrain "/root/autodl-tmp/yiming/ikemura_ws/PDNE_VPT_V2/PDNE/experiments/230928_063056_model-4-1-0/model_00050.pt" \

                # --resume \
                # --pretrain /root/autodl-tmp/yiming/ikemura_ws/PDNE_VPT/PDNE/experiments/230919_040259_model-2/model_00020.pt \
                