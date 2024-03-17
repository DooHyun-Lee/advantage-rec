import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from scipy.sparse import csr_matrix

from model import AdvFunc
from tqdm import tqdm
from typing import List, Set
import math
import logging
import os

log = logging.getLogger(__name__)

class Trainer:
    def __init__(self, total_seq:List[int], total_item:Set[int], model:AdvFunc, train_loader:DataLoader, 
                 valid_loader:DataLoader, test_loader:DataLoader, model_name:str, args):
        self.total_seq = total_seq
        self.total_item = total_item
        self.adv_func = model
        self.adv_func.to(args.device)
        betas = (args.beta1, args.beta2)
        self.optim = torch.optim.Adam(self.adv_func.parameters(), lr=args.lr,
                                      betas=betas, weight_decay=args.weight_decay)

        self.trainLoader = train_loader
        self.validLoader = valid_loader
        self.testLoader = test_loader
        self.generate_matrix()

        self.model_name = model_name
        self.epochs = args.epochs
        self.device = args.device
        self.temperature = args.temperature
        self.adv_weight = args.adv_weight
        self.cl_weight = args.cl_weight
        self.early_stop = args.early_stop
        self.save_path = args.save_path

    def generate_matrix(self):
        '''
        generate matrix for validation, test
        '''
        num_users = len(self.total_seq)
        num_items = len(self.total_item) + 2
        row_val, col_val, data_val = [],[],[]
        row_test, col_test, data_test = [],[],[]
        for i, cur_seq in enumerate(self.total_seq):
            row_val.extend([i]*len(cur_seq[:-2]))
            col_val.extend(cur_seq[:-2])
            data_val.extend([1]*len(cur_seq[:-2]))
            row_test.extend([i]*len(cur_seq[:-1]))
            col_test.extend(cur_seq[:-1])
            data_test.extend([1]*len(cur_seq[:-1]))

        def to_csr(row:List[int], col:List[int], data:List[int], 
                   num_users:int, num_items:int) -> csr_matrix:
            row, col, data = np.array(row), np.array(col), np.array(data)
            matrix = csr_matrix((data, (row,col)), shape=(num_users, num_items))
            return matrix
        
        self.val_matrix = to_csr(row_val, col_val, data_val,
                                 num_users, num_items)
        self.test_matrix = to_csr(row_test, col_test, data_test, 
                                  num_users, num_items)

    def regret_preference(self, cur_seq:torch.Tensor, win_label:torch.Tensor, 
                          lose_label:torch.Tensor)->torch.Tensor:
        '''
        maximize regret based preference model  
        '''
        adv_pos, adv_neg, target_mask = self.adv_func.get_adv(cur_seq, win_label, lose_label)
        loss = torch.sum(-F.logsigmoid(adv_pos-adv_neg)*target_mask) / torch.sum(target_mask)
        return loss

    def cl_regularizer(self, cl_data:List[torch.Tensor]) -> torch.Tensor:
        '''
        regularize model with cl objective
        heavly rely on https://github.com/YChen1993/CoSeRec/blob/main/src/modules.py
        '''
        bsize = cl_data[0].size(0)
        cl_concat = torch.cat(cl_data, dim=0)
        cl_concat = cl_concat.to(self.device)
        cl_concat_encoded = self.adv_func.get_vs(cl_concat)
        cl_concat_flatten = cl_concat_encoded.view(bsize*2, -1)
        cl_concat_list = torch.split(cl_concat_flatten, bsize)
        cl_sample_one, cl_sample_two = cl_concat_list
        mat11 = torch.matmul(cl_sample_one, cl_sample_one.T) / self.temperature
        mat12 = torch.matmul(cl_sample_one, cl_sample_two.T) / self.temperature
        mat22 = torch.matmul(cl_sample_two, cl_sample_two.T) / self.temperature
        mat11[range(bsize), range(bsize)] = float('-inf')
        mat22[range(bsize), range(bsize)] = float('-inf')
        #       mat2 mat1
        #       --------- 
        # mat1 |
        # mat2 |
        raw_scores1 = torch.cat([mat12, mat11], dim=-1)
        raw_scores2 = torch.cat([mat22, mat12.T], dim=-1)
        logits = torch.cat([raw_scores1, raw_scores2], dim=0)
        labels = torch.arange(2*bsize, dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss 

    def train(self):
        best_score = None
        stopping_cnter = 0
        self.adv_func = self.adv_func.to(self.device)
        for epoch in range(self.epochs):
            if stopping_cnter > self.early_stop:
                break
            adv_avg_loss = 0.0
            cl_avg_loss = 0.0
            tot_avg_loss = 0.0
            self.adv_func.train()
            for (adv_data, cl_data) in tqdm(self.trainLoader, total=len(self.trainLoader)):
                adv_data = tuple(map(lambda x: x.to(self.device), adv_data))
                index, cur_seq, win_label, lose_label, last_label = adv_data
                adv_loss = self.regret_preference(cur_seq, win_label, lose_label)
                cl_loss = self.cl_regularizer(cl_data)
                tot_loss = adv_loss * self.adv_weight + cl_loss * self.cl_weight

                self.optim.zero_grad()
                tot_loss.backward()
                self.optim.step()

                adv_avg_loss += adv_loss.item()
                cl_avg_loss += cl_loss.item()
                tot_avg_loss += tot_loss.item()

            log.info(f'epoch: {epoch} || adv_avg_loss: {adv_avg_loss/len(self.trainLoader):.4f} |'  
                    f'cl_avg_loss: {cl_avg_loss/len(self.trainLoader):.4f} | ' 
                    f'tot_avg_loss: {tot_avg_loss/len(self.trainLoader):.4f} |')

            _, ndcgs = self.test(eval_type='valid')
            if not best_score:
                best_score = ndcgs[-1]
                self.save()
                log.info(f'saving model after new best score: {best_score:.4f}')
            else:
                if best_score > ndcgs[-1]:
                    stopping_cnter += 1
                    log.info(f'current early stop cnter: {stopping_cnter}')
                else:
                    best_score = ndcgs[-1]
                    stopping_cnter = 0
                    self.save()
                    log.info(f'saving model after new best score: {best_score:.4f}')

    def test(self, eval_type:str) -> tuple[List[float]]:
        assert eval_type in ['valid', 'test']
        if eval_type == 'valid':
            dataLoader = self.validLoader
            matrix = self.val_matrix
        elif eval_type == 'test':
            dataLoader = self.testLoader
            matrix = self.test_matrix
            # load best saved model
            self.load()
            
        self.adv_func.eval()
        preds = []
        answers = []
        for adv_data in tqdm(dataLoader, total=len(dataLoader)):
            adv_data = tuple(map(lambda x: x.to(self.device), adv_data))
            index, cur_seq, win_label, lose_label, last_label = adv_data
            adv = self.adv_func.get_all_adv(cur_seq)
            # reference : https://github.com/YChen1993/CoSeRec/blob/main/src/trainers.py
            '''
            index = index.cpu().numpy()
            adv = adv.detach().cpu().numpy()
            answer = last_label.detach().cpu().numpy()
            # exclude items already bought
            adv[matrix[index].toarray()>0] = 0
            batch_idx = np.expand_dims(np.arange(len(adv)), axis=1)
            top_20_idx = np.argpartition(adv, -20)[:, -20:]
            top_20_val = adv[batch_idx, top_20_idx]
            top_20_val_idx = np.argsort(top_20_val, axis=1)[::-1]
            pred = top_20_idx[batch_idx, top_20_val_idx]
            preds.append(pred)
            answers.append(answer)
            '''
            adv = adv.cpu().data.numpy().copy()
            index = index.cpu().numpy()
            adv[matrix[index].toarray()>0] = 0
            ind = np.argpartition(adv, -20)[:, -20:]
            arr_ind = adv[np.arange(len(adv))[:, None], ind]
            arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(adv)), ::-1]
            pred = ind[np.arange(len(adv))[:, None], arr_ind_argsort]
            preds.append(pred)
            answers.append(last_label.cpu().data.numpy())

        answers = np.concatenate(answers, axis=0)
        preds = np.concatenate(preds, axis=0)
        hit_rates, ndcgs = self.metric_scores(answers, preds)
        log.info(f'---------{eval_type} result! ----------')
        log.info(f'HIT@5: {hit_rates[0]:.4f}|  NDCG@5: {ndcgs[0]:.4f}| '
                 f'HIT@10: {hit_rates[1]:.4f}|  NDCG@10: {ndcgs[1]:.4f}| '
                 f'HIT@20: {hit_rates[2]:.4f}|  NDCG@20: {ndcgs[2]:.4f}')
        return hit_rates, ndcgs
    
    def metric_scores(self, answers:np.array, preds:np.array) -> tuple[List[float]]:
        def hit_rate(answers:np.array, preds:np.array, k:int)-> float:
            hit_cnt = 0
            assert len(answers) == len(preds)
            for answer, pred in zip(answers, preds):
                if answer in pred[:k]:
                    hit_cnt += 1
            return float(hit_cnt / len(answers))

        def ndcg(answers:np.array, preds:np.array, k:int) -> float:
            ndcg = 0
            assert len(answers) == len(preds)
            for answer, pred in zip(answers, preds):
                idcg = (1.0/math.log(1+1, 2))
                dcg = sum([int(pred[j] in answer) / math.log(j+2,2) for j in range(k)])
                ndcg += dcg/idcg
            return ndcg / float(len(answers))

        hit_rates, ndcgs = [], []
        for k in [5, 10, 20]:
            hit_rates.append(hit_rate(answers, preds, k))
            ndcgs.append(ndcg(answers, preds, k))

        return hit_rates, ndcgs

    def save(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        save_path = os.path.join(self.save_path, self.model_name)
        torch.save(self.adv_func.state_dict(), save_path)

    def load(self):
        load_path = os.path.join(self.save_path, self.model_name)
        self.adv_func.load_state_dict(torch.load(load_path))