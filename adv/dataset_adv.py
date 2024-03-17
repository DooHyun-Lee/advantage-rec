import torch
from torch.utils.data import Dataset
from typing import List, Set, Tuple, Any
import random
import copy
from augmentation import Augmentation

def total_sequence(path:str) -> Tuple[List[List[int]], Set[int]]:
    '''
    return total interaction history, item set
    '''
    total_sequences = []
    total_item = set()
    contain_user = 'gpt2' in path
    with open(path, 'r') as f:
        for line in f:
            cur_items = line.strip().split(' ')
            cur_items = list(map(int, cur_items))
            if contain_user:
                cur_items = cur_items[1:]
            total_sequences.append(cur_items)
            total_item.update(cur_items)
    return total_sequences, total_item 

def sample(exclude_set:Set[int], item_set:List[int]) -> int:
    item = random.choice(item_set)
    while item in exclude_set:
        item = random.choice(item_set)
    return item

def pad(input:List[int], max_len:int) -> List[int]:
    pad_len = max_len - len(input)
    input = [0] * pad_len + input
    input = input[-max_len:]
    return input

class AdvDataset(Dataset):
    def __init__(self, args, total_sequence:List[List[int]], total_item:Set[int], train='train'):
        self.total_sequence = total_sequence
        self.total_item = total_item
        self.total_item_list = list(total_item)
        self.total_item_nums = len(total_item) + 2  # padding + masking
        self.train = train
        self.max_len = args.max_len
        self.augmentations = Augmentation(tao=args.tao, gamma=args.gamma, beta=args.beta)

    def return_item_nums(self):
        return self.total_item_nums

    def cl_dataset(self, cur_sequence:List[int]) -> List[List[int]]:
        '''
        cl objective for regularization purpose
        https://github.com/salesforce/ICLRec/blob/master/src/datasets.py
        '''
        cl_sequences = []
        for _ in range(2):
            augmented_cur_sequence = self.augmentations(cur_sequence)
            augmented_cur_sequence = pad(augmented_cur_sequence, self.max_len)
            cl_sequences.append(augmented_cur_sequence)
        return list(map(lambda x: torch.tensor(x, dtype=torch.long), cl_sequences))

    def preference_dataset(self, index:int, exclude:List[int], 
            cur_sequence:List[int], win_label:List[int], last_label:List[int]) -> Tuple[torch.Tensor]:
        '''
        used for both advantage func training and validation, test 
        '''
        exclude_set = set(exclude)
        lose_label = []
        # TODO : add differenet types of lose label generation
        for _ in win_label:
            lose_label.append(sample(exclude_set, self.total_item_list))

        cur_sequence = pad(cur_sequence, self.max_len)
        win_label = pad(win_label, self.max_len)
        lose_label = pad(lose_label, self.max_len)

        preference_data = (index, cur_sequence, win_label, lose_label, last_label)
        return tuple(map(lambda x: torch.tensor(x, dtype=torch.long), preference_data))

    def __getitem__(self, index) -> Any:
        sequence = self.total_sequence[index]
        if self.train == "train":
            cur_sequence = sequence[:-3]
            win_label = sequence[1:-2]
            last_label = sequence[-2]
            cl_data = self.cl_dataset(cur_sequence)
            adv_data = self.preference_dataset(index=index, exclude=sequence,
                                               cur_sequence=cur_sequence, win_label=win_label,
                                               last_label=last_label)
            return (adv_data, cl_data)

        elif self.train == "valid":
            cur_sequence = sequence[:-2]
            win_label = sequence[1:-1]
            last_label = [sequence[-2]]
            adv_data = self.preference_dataset(index=index, exclude=sequence,
                                               cur_sequence=cur_sequence, win_label=win_label,
                                               last_label=last_label)
            return adv_data
        else:
            cur_sequence = sequence[:-1]
            win_label = sequence[1:]
            last_label = [sequence[-1]]
            adv_data = self.preference_dataset(index=index, exclude=sequence,
                                               cur_sequence=cur_sequence, win_label=win_label,
                                               last_label=last_label)
            return adv_data

    def __len__(self):
        return len(self.total_sequence)