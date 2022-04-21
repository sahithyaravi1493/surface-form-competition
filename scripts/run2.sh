CUDA_VISIBLE_DEVICES=1 python -u score.py obqa  --model gpt2 --batch 128
CUDA_VISIBLE_DEVICES=1 python -u score.py hellaswag  --model gpt2 --batch 128