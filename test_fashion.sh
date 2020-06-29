
python -u test.py \
--dataroot ./dataset/fashion_data/ --dataset fashion --phase test --norm instance --batchSize 1 --ngf 64 --resize_or_crop no --gpu_ids 0,1 --BP_input_nc 18 --no_flip --pairLst ./dataset/fashion_data/fashion-resize-pairs-test.csv --display_id 0 --which_epoch latest --how_many 100000 --G_n_downsampling 5 --results_dir ./results --model PoseStyleNet \
--name fashion_APS --which_model_netG APS --dataset_mode keypoint