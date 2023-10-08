#chmod +x test.sh
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=4318 --use_env basicsr/test.py -opt ./options/test/S-C_dark_1_d100_wall.yml --launcher pytorch


