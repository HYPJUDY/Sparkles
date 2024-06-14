CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 evaluate_I4.py \
--cfg-path /mnt/localdata/Users/yupanhuang/code/Sparkles/eval_configs/sparkles_eval.yaml \
--gpu-id 0 --batch-image 24 --i4-dir /mnt/localdata/Users/yupanhuang/data/I4-Core \
--dataset NLVR2 --result-dir /mnt/localdata/Users/yupanhuang/models/Sparkles/eval_I4/pretrained_sparkleschat_7b \
--sparkles_root /mnt/localdata/Users/yupanhuang/data/Sparkles