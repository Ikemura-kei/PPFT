#! /bin/bash

# lei7 
# Modality promper
export CUDA_VISIBLE_DEVICES=4,5
python main.py --dir_data /root/autodl-tmp/yiming/datasets/polar_hammer \
                --data_name HAMMER \
                --data_txt /root/autodl-tmp/yiming/ikemura_ws/PDNE/data_paths/hammer_MODE.txt \
                --gpus 4,5 \
                --loss 1.0*L1+1.0*L2 \
                --batch_size 14 \
                --epochs 250 \
                --log_dir ./experiments/ \
                --save model-24 \
                --model ISDPromptFinetuneV2Freeze \
                --completionformer_mode rgbd \
                --pre_pvt \
                --pre_res \
                --save_full \
                --warm_up \
                --port 29502 \
                --lr 0.00105 \
                --pretrained_completionformer /root/autodl-tmp/yiming/PDNE/pretrained/comp/NYUv2.pt \
                --use_pol \
                --pol_rep leichenyang-7 \
                --resume --pretrain /root/autodl-tmp/yiming/PDNE_all/PDNE_ISD_PROMPT_FINETUNE/experiments/231101_020155_model-24/model_00022.pt