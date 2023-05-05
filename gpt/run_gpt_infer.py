
from cProfile import label
from pickle import Unpickler
import numpy as np
import time
import os
#os.environ["PVR_GPUIDX"] = str(1)
os.environ["MTGPU_MAX_MEM_USAGE_GB"] = "28"
import logging
import math
import torch
from arg import get_arg
from evaluation import evaluate
from logger import init_logger
try:
    import torch_musa
except ImportError:
    pass

from torch import nn
from torch.utils.data import Dataset, DataLoader
from dataset import WebtextDataset
from transformers import GPT2Config, AutoTokenizer, GPT2LMHeadModel
from torch.profiler import profile, record_function, ProfilerActivity, schedule

import psutil

def warmup_lambda_lr(train_step):
    if train_step<args.num_warmup_steps:
        return args.learning_rate * train_step / args.num_train_steps
    else:
        return args.learning_rate

if __name__ == '__main__':
    args = get_arg()

    if args.log_file is None:
        args.log_file = f'./log/{args.exp_name}/log_{time.time()}.txt'
        if not os.path.exists(f'./log/{args.exp_name}'): os.mkdir(f'./log/{args.exp_name}')
    if args.output_dir is None:
        args.output_dir = f'./checkpoint/{args.exp_name}'
        if not os.path.exists(f'./checkpoint/{args.exp_name}'): 
            os.mkdir(f'./checkpoint/{args.exp_name}')

    logger = init_logger(args)

    if args.input_eval_file is None:
        logger.info(f'no evaluation file give use the train file {args.input_train_file} instead')
        args.input_eval_file = args.input_train_file

    data = WebtextDataset(args.input_train_file, tokenize_path=args.tokenize_path) #1G
    eval_loader = DataLoader(data, batch_size=args.eval_batch_size, shuffle=True, pin_memory=False)

    configuration = GPT2Config()
    print(configuration)
    print('-' * 10)
    model = GPT2LMHeadModel(configuration)
    numel = sum([p.numel() for p in model.parameters()])
    print(f'model parameter nueml {numel}')
    # model.gradient_checkpointing_enable()    
    
    model = model.to(args.device)

    if args.init_checkpoint is not None:
        logger.info(f'load states dict from {args.init_checkpoint}')
        states = torch.load(args.init_checkpoint)
        logger.info(f'init model states...')
        model.load_state_dict(states['model_states'])

    
    # start to eval
    model.eval()
    total_loss = 0
    eval_step = 0
    start = 0
    start = time.time()
    with torch.no_grad():
        for features in eval_loader:
            if eval_step == 1:
              eval_start = time.time()
            input_ids = features[0]['input_ids']
            attention_mask = features[0]['attention_mask']
            labels = features[1]
            input_ids = input_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            labels = labels.to(args.device)
            
            CausalLMOutput = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = CausalLMOutput.loss
            total_loss += loss.item()
            print(f'eval_step:{eval_step}')
            eval_step += 1
            if eval_step == 30:
                break

    total_loss /= eval_step
    avg_eval_time = (time.time() - eval_start) / (eval_step-1)
    avg_infer_fps = args.eval_batch_size / avg_eval_time
    logger.info(f'evaluate loss is {total_loss:.4f}, ppl is {math.exp(total_loss)}, has cost {(time.time() - start):.3f}, avg_eval_time is: {avg_eval_time:.3f}, infer_fps:{avg_infer_fps:.3f}') 
 

