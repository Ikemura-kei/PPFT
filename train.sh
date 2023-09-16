python main.py --data_root ../datasets/hammer_polar_mini \
--data_name SPW2 --pre_pvt --save_full --with_norm \
--val_freq 10 --save_freq 10 --direct_cat --slice \
--port 29512 --sparse_dir spw2_sparse_hollow_old \
--save gray_hollow_old_norm --netinput onlyiun_pol_vd \
--mode grayd --split_json ../data_json/nyu.json \
--gpus 6,7 --loss 1.0*L1+1.0*L2 --batch_size 4 \
--milestones 100 200 300 400 500 \
--epochs 600 --log_dir ../experiments_new/ 