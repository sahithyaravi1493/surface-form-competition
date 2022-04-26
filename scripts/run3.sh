CUDA_VISIBLE_DEVICES=2 python -u score.py race-h  --model gpt2 --batch 128 --abstraction_method 'both'
CUDA_VISIBLE_DEVICES=2 python -u score.py race-m  --model gpt2 --batch 128 --abstraction_method 'both'
CUDA_VISIBLE_DEVICES=2 python -u score.py rte  --model gpt2 --batch 128 --abstraction_method 'both'
