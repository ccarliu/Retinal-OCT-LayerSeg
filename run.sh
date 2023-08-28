# for dme dataset:

CUDA_VISIBLE_DEVICES=0 python train_dme.py --gpu 0 --root yourdatapath --batch_size 4 --name test --weight_decay 0.0005 --lr 0.005 --patch_size 48
CUDA_VISIBLE_DEVICES=0 python test_dme.py --gpu 0 --root yourdatapath --ck_name dme_ck2.t7 --batch_size 4 --name test --patch_size 48
