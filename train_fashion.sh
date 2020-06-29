
python -u train.py \
--dataroot ./dataset/fashion_data/ --dataset fashion --lambda_GAN 5 --lambda_A 1 --lambda_B 1 --n_layers 3 --norm instance --pool_size 0 --resize_or_crop no --BP_input_nc 18 --no_flip --pairLst ./dataset/fashion_data/fashion-resize-pairs-train.csv --L1_type l1_plus_perL1 --n_layers_D 3 --with_D_PP 1 --with_D_PB 1 --display_id 0 --model PoseStyleNet --print_freq 10 --save_latest_freq 100 --save_epoch_freq 100 --continue_train --which_epoch latest \
--niter 400 --niter_decay 400 --lr 0.0002 --G_n_downsampling 5 \
--gpu_ids 0,1,2,3,4,5,6,7 --batchSize 32 --ngf 64 \
--name fashion --which_model_netG APS --dataset_mode keypoint
