# CUDA_VISIBLE_DEVICES=0 python train_dme.py --gpu 0 --batch_size 2 --name test --weight_decay 0.0005 --lr 0.005 --patch_size 48
CUDA_VISIBLE_DEVICES=0 python test_dme.py --gpu 0 --ck_name refinedgdweightx0_wdecay00001_lr001_p48_nopretrain_056_fold0_2.131%.t7 --batch_size 2 --name test --patch_size 48
