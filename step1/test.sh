CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=4201 basicsr/test.py -opt ./options/test/Supermodel.yml --launcher pytorch
