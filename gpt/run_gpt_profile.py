
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

def evaluation(model, args, loss_fn):
    _ = model.eval()
    
    total_loss = 0
    total_n = 0

    eval_data = BertDataset(args.input_eval_file)
    evalloader = DataLoader(eval_data, batch_size=args.eval_batch_size, shuffle=False)
    eval_step = 0

    with torch.no_grad():
        for features in evalloader:
            logits_mlm, logits_nsp = _model_forward(model, features, args)
            loss = _get_total_loss(loss_fn, logits_mlm, logits_nsp, features, args)

            n = logits_mlm.shape[0]
            total_loss += loss * n
            total_n += n

            eval_step += 1
            if eval_step>=args.max_eval_steps:
                break

    _ = model.train()
    return total_loss / (total_n+1e-9)


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
    # len_data = len(data)
    # train_data = data[ : round(len_data * 0.9)]
    # eval_data = data[round(len_data * 0.9) : ]
    train_loader = DataLoader(data, batch_size=args.train_batch_size, shuffle=True)
    # eval_loader = DataLoader(eval_data, batch_size=args.eval_batch_size, shuffle=True)

    configuration = GPT2Config()
    # configuration.n_embd = 1536
    # configuration.n_head = 24
    # configuration.n_layer = 1
    # configuration.vocab_size = 21128
    # pdrop = 0
    # configuration.resid_pdrop = pdrop
    # configuration.embd_pdrop = pdrop
    # configuration.attn_pdrop = pdrop
    #configuration.activation_function = 'gelu'
    print(configuration)
    # exit()
    print('-' * 10)
    model = GPT2LMHeadModel(configuration) #.to("mtgpu")
    #print(model.transformer.wte.weight.data_ptr())
    #print(model.lm_head.weight.data_ptr())
    #print(id(model.transformer.wte.weight))
    #print(id(model.lm_head.weight))
    # m_d = model.state_dict()
    # torch.save(m_d, f'./cai_gpt_{configuration.n_layer}.pt')
    # exit()

    numel = sum([p.numel() for p in model.parameters()])
    print(f'model parameter nueml {numel}')
    # model.gradient_checkpointing_enable()    
    
    model = model.to(args.device)
    #print('-' * 10)
    #print(model.transformer.wte.weight.data_ptr())
    #print(model.lm_head.weight.data_ptr())
    #print(id(model.transformer.wte.weight))
    #print(id(model.lm_head.weight))
    #exit()
    optimizer = torch.optim.Adam(params=model.parameters(), lr = args.learning_rate)

    _ = model.train()

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda_lr)

    if args.init_checkpoint is not None:
        logger.info(f'load states dict from {args.init_checkpoint}')
        states = torch.load(args.init_checkpoint)
        logger.info(f'init model states...')
        model.load_state_dict(states['model_states'])
        logger.info(f'init optimizer states...')
        optimizer.load_state_dict(states['optimizer_states'])

    STEPS = 0
    epoch = 0
    forward_time = 0
    backward_time = 0
    optimizer_time = 0
    all_forward_time = 0
    all_backward_time = 0
    all_optimizer_time = 0
    all_time = 0

    print('start_train')
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True,
                 schedule=torch.profiler.schedule(
                 wait=10,
                 warmup=1,
                 active=1,
                 repeat=1),
                 on_trace_ready=torch.profiler.tensorboard_trace_handler('./profile/gpt')) as p:
    
        while STEPS < args.num_train_steps:
       
            for features in train_loader:
                optimizer.zero_grad()

                input_ids = features[0]['input_ids']
                attention_mask = features[0]['attention_mask']
                labels = features[1]
                
                input_ids = input_ids.to(args.device)
                attention_mask = attention_mask.to(args.device)
                labels = labels.to(args.device)

                start = time.time()

                CausalLMOutput = model(input_ids, attention_mask=attention_mask, labels=labels)

                forward_time = time.time() - start
                all_forward_time += forward_time

                loss = CausalLMOutput.loss
                loss.backward()

                backward_time = time.time() - forward_time - start
                all_backward_time += backward_time
            
                optimizer.step()

                optimizer_time = time.time() - backward_time - forward_time - start
                all_optimizer_time += optimizer_time
                
                all_time += forward_time + backward_time + optimizer_time
            
                STEPS += 1

                if STEPS % args.log_steps == 0:
                    try:
                        logger.info(f'epoch {epoch}, step {STEPS}, loss is {loss.item():.4f}, ppl is {math.exp(loss.item())}')
                    except:
                        exit()
                    logger.info(f'epoch {epoch}, step {STEPS}, loss is {loss.item():.4f}')
                    logger.info(f'forward has cost {(all_forward_time / args.log_steps) :.3f}')
                    logger.info(f'backward has  cost {(all_backward_time / args.log_steps) :.3f}')
                    logger.info(f'optimizer has cost {(all_optimizer_time / args.log_steps) :.3f}')
                    logger.info(f'all_time is {(all_time / args.log_steps) :.3f}')
                    logger.info(f'memory has used {psutil.Process().memory_info().rss / 1024**2:.2f} MB')
                    logger.info('====================><====================')
                    all_forward_time, all_backward_time, all_optimizer_time, all_time = 0, 0, 0, 0

                    if STEPS % args.save_checkpoints_steps == 0:
                        if args.output_dir is not None:
                            output_checkpoint_name = args.output_dir + f'/checkpoints_{STEPS}_{time.time()}.pt'
                            logger.info(f'save model and optimizer to {output_checkpoint_name}')
                            ckpt = {
                                    'model_states':model.state_dict(), 
                                    'optimizer_states':optimizer.state_dict()
                                }
                            torch.save(ckpt, output_checkpoint_name)
                            #model.to("mtgpu")
                        else:
                            logger.info(f'not save checkpoint.')
                    
                    # if STEPS % args.evaluation_steps == 0:
                    #     evaluate(model, eval_loader, args, logger)

                    # if STEPS % args.evaluation_steps == 0:
                    #     eval_loss = evaluation(model, args, loss_fn).cpu().item()
                    #     logger.info(f'step {STEPS} evaluation loss: {eval_loss}')

                    if STEPS >= args.num_train_steps:
                        break
                p.step()
            epoch += 1
            
 
