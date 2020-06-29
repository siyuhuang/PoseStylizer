
python -u test.py \
--dataroot ./dataset/market_data --phase test --norm batch --batchSize 1 --resize_or_crop no --BP_input_nc 18 --no_flip --how_many 100000 --pairLst ./dataset/market-pairs-test.csv --display_id 0 --which_epoch latest --gpu_ids 0,1 --G_n_downsampling 4 --ngf 64 --model PoseStyleNet --results_dir ./results \
--name market_APS --which_model_netG APS --dataset_mode keypoint
