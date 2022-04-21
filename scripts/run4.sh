CUDA_VISIBLE_DEVICES=3 python -u score.py sst-2  --model gpt2 --batch 256
CUDA_VISIBLE_DEVICES=3 python -u score.py sst-5  --model gpt2 --batch 256 
CUDA_VISIBLE_DEVICES=3 python -u score.py storycloze --model gpt2 --batch 256
CUDA_VISIBLE_DEVICES=3 python -u score.py trec  --model gpt2 --batch 128
