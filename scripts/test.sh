#! /bin/bash

python main.py --dir_data ./data/hammer_polar \
                --data_name HAMMER \
                --data_txt ./data_paths/hammer_MODE.txt \
                --gpus 0 \
                --loss 1.0*L1+1.0*L2 \
                --log_dir ./experiments/ \
                --save ppft_final \
                --model PPFT \
                --completionformer_mode rgbd \
                --pre_pvt \
                --pre_res \
                --pretrained_completionformer ./ckpts/NYUv2.pt \
                --use_pol \
                --pol_rep leichenyang-7 \
                --test_only \
                --data_percentage 1 \
                --pretrain_list_file ./ckpts/ppft_final.txt \