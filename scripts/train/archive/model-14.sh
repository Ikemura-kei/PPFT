#! /bin/bash

# lei7 

python main.py --dir_data /root/autodl-tmp/yiming/datasets/polar_hammer \
                --data_name HAMMER \
                --data_txt /root/autodl-tmp/yiming/ikemura_ws/PDNE/data_paths/hammer_MODE.txt \
                --gpus 2,3 \
                --loss 1.0*L1+1.0*L2 \
                --batch_size 14 \
                --epochs 250 \
                --log_dir ./experiments/ \
                --save model-16 \
                --model ISDPromptFinetune \
                --completionformer_mode rgbd \
                --pre_pvt \
                --pre_res \
                --save_full \
                --warm_up \
                --port 29501 \
                --lr 0.001 \
                --pretrained_completionformer /root/autodl-tmp/yiming/PDNE/pretrained/comp/NYUv2.pt \
                --use_pol \
                --pol_rep leichenyang-7 \
                --resume --pretrain /root/autodl-tmp/yiming/PDNE_all/PDNE_ISD_PROMPT_FINETUNE/experiments/231024_211417_model-16/model_00014.pt