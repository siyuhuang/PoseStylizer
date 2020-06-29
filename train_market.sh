
python -u train.py \
--dataroot ./dataset/market_data/ --lambda_GAN 5 --lambda_A 10  --lambda_B 10 --no_lsgan --n_layers 3 --norm batch --no_flip --resize_or_crop no --BP_input_nc 18 --pairLst ./dataset/market_data/market-pairs-train.csv --n_layers_D 3 --with_D_PP 1 --with_D_PB 1 --display_id 0 --continue_train --L1_type l1_plus_perL1 --model PoseStyleNet --print_freq 10 --save_latest_freq 50 --save_epoch_freq 100 --continue_train --which_epoch latest \
--niter 400 --niter_decay 400 --lr 0.0002 --G_n_downsampling 4 \
--gpu_ids 0,1,2,3,4,5,6,7 --batchSize 192 --ngf 64 \
--name market --which_model_netG APS --dataset_mode keypoint


