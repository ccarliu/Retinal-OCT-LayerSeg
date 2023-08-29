# for dme dataset:

#CUDA_VISIBLE_DEVICES=0 python3 train_dme.py --gpu 0 --data_dir yourdatapath --batch_size 4 --name test --weight_decay 0.0005 --lr 0.005 --patch_size 48
CUDA_VISIBLE_DEVICES=0 python3 test_dme.py --gpu 0 --data_dir yourdatapath --ck_name dme_ck1.t7 --batch_size 4 --name test --patch_size 48
