CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=4201 --use_env basicsr/train.py -opt options/train/Supermodel.yml --launcher pytorch


