import subprocess

cmd = 'python run_gpt.py \
        --input_train_file ./data/12.txt \
        --learning_rate 3.0e-4 \
        --train_batch_size 16 \
        --eval_batch_size 16 \
        --evaluation_steps 1 \
        --device musa \
        --tokenize_path /home/kangchen/Gpt/roberta \
        --log_steps 5 \
        --save_checkpoints_steps 10000'.split() #    --device mtgpu \

print(cmd)

try:
    res = subprocess.call(cmd)
except KeyboardInterrupt:
    print('Interupt')

