CUDA_VISIBLE_DEVICES=2 python -u score.py race-h  --model gpt2 --batch 128
CUDA_VISIBLE_DEVICES=2 python -u score.py race-m  --model gpt2 --batch 128
CUDA_VISIBLE_DEVICES=2 python -u score.py rte  --model gpt2 --batch 128
