import torch
import torch.nn as nn
import numpy as np
import transformers
import os
from typing import Dict, Union, List
import random
import logging 
from collections import defaultdict

from dataset_gpt2 import get_batch_iterator, seq_generator
from utils import move_batch_to_device, formatted_dict

log = logging.getLogger(__name__)

def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
    assert logits.shape[:-1] == labels.shape
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :] 
    loss_mask = (labels != -100)

    labels[labels == -100] = 0
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    return (per_token_logps * loss_mask).sum(-1)

class Trainer():
    def __init__(self, policy:nn.Module, args, seed:int):
        self.seed = seed
        self.args = args

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_type)
        new_tokens = []
        item_set = set()
        dataset_name = args.dataset_name + '.txt'
        data_path = os.path.join('../data', dataset_name)
        with open(data_path, 'r') as f:
            for line in f:
                data = line.strip()
                data = data.split(' ')
                items = data[1:]
                for item in items:
                    item_set.add(item)
        for item in item_set:
            new_tokens.append(f'item_{item}')
        self.tokenizer.add_tokens(new_tokens)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.policy = policy
        self.policy.resize_token_embeddings(len(self.tokenizer))

        train_seq, valid_seq, test_seq = seq_generator(data_path)
        self.train_iterator = get_batch_iterator(self.tokenizer, train_seq, 
                                                 args.batch_size, shuffle=True, 
                                                 max_length=args.max_length, n_epochs=args.n_epochs,
                                                 seed=args.seed)
    
    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], train=True):
        metrics = {}
        train_test = 'train' if train else 'valid'
        policy_target_logits = self.policy(batch['target_input_ids'], 
                                           attention_mask=batch['target_attention_mask']).logits.to(torch.float32)
        policy_target_logps = _get_batch_logps(policy_target_logits, batch['target_labels'])
        losses = -policy_target_logps
        metrics[f'loss_{train_test}'] = losses.detach().cpu().numpy().tolist()
        return losses.mean(), metrics

    def clip_gradient(self):
        return nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.max_grad_norm).item()

    def train(self):
        self.policy = self.policy.to(self.args.device)
        self.optimizer = getattr(torch.optim, self.args.optimizer)(self.policy.parameters(), self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step+1) / self.args.warmup_steps+1))

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.example_counter = 0
        self.batch_counter = 0
        
        for batch in self.train_iterator:
            self.policy.train()
            batch_metrics = defaultdict(list)
            for microbatch_idx in range(self.args.gradient_accumulation_steps):
                micro_batch = move_batch_to_device(batch, microbatch_idx, self.args.gradient_accumulation_steps,
                                                   self.arg.device)
                loss, metrics = self.get_batch_metrics(micro_batch)
                (loss / self.args.gradient_accumulation_steps).backward()
                for k, v in metrics.items():
                    batch_metrics[k].extend(v)
                
            grad_norm = self.clip_gradient()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            batch_metrics['grad_norm'].append(grad_norm)
            self.batch_counter += 1
            self.example_counter += self.args.batch_size

            mean_train_metrics = {k:sum(v) / len(v) for k, v in batch_metrics.items()}
            log.info(f'train metrics at {self.example_counter} examples : {formatted_dict(mean_train_metrics)}')

            if self.example_counter % self.args.eval_every == 0 and self.example_counter > 0:
                self.save(self.args.save_path)

    def save(self, path:str):
        if not os.path.exists(path):
            os.makedirs(path)
        model_save_path = path + self.args.dataset_name + '_policy.pt'
        optimizer_save_path = path + self.args.dataset_name + '_optimizer.pt'
        scheduler_save_path = path + self.args.dataset_name + '_scheduler.py'
        model_state_dict = self.policy.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        scheduler_state_dict = self.scheduler.state_dict()
        torch.save({
            'step_idx' : self.example_counter,
            'state' : model_state_dict
        }, model_save_path)
        del model_state_dict
        torch.save({
            'state_idx' : self.example_counter,
            'state' : optimizer_state_dict
        }, optimizer_save_path)
        del optimizer_state_dict
        torch.save({
            'state_idx': self.example_counter,
            'state' : scheduler_state_dict
        }, scheduler_save_path)