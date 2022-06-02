# CUDA_VISIBLE_DEVICES=0 python -u score.py arc-challenge  --model gpt2 --batch 32 --abstraction_method 'both'
# CUDA_VISIBLE_DEVICES=0 python -u score.py arc-easy  --model gpt2 --batch 32  --abstraction_method 'both'
# CUDA_VISIBLE_DEVICES=0 python -u score.py cqa  --model gpt2 --batch 32 --abstraction_method 'both'
CUDA_VISIBLE_DEVICES=0 python -u score.py copa  --model gpt2 --batch 16 --abstraction_method 'both'
# CUDA_VISIBLE_DEVICES=0 python -u score.py cqa --model gpt2 --batch 32 --abstraction_method 'both'
# CUDA_VISIBLE_DEVICES=0 python -u score.py rte  --model gpt2 --batch 32 --abstraction_method 'both'
# CUDA_VISIBLE_DEVICES=0 python -u score.py race-m --model gpt2 --batch 32 --abstraction_method 'both'







