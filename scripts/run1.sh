CUDA_VISIBLE_DEVICES=0 python -u score.py arc-challenge  --model gpt2 --batch 128 --abstraction_method 'both'
CUDA_VISIBLE_DEVICES=0 python -u score.py arc-easy  --model gpt2 --batch 128  --abstraction_method 'both'
CUDA_VISIBLE_DEVICES=0 python -u score.py cb  --model gpt2 --batch 64 --abstraction_method 'both'
CUDA_VISIBLE_DEVICES=0 python -u score.py copa  --model gpt2 --batch 128 --abstraction_method 'both'
CUDA_VISIBLE_DEVICES=0 python -u score.py cqa --model gpt2 --batch 128 --abstraction_method 'both'









