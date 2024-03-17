import copy
import random
from typing import List

'''
much of the augmentation details rely on ICLRec
https://github.com/salesforce/ICLRec/blob/master/src/data_augmentation.py
'''

class Augmentation():
    """Randomly pick one data augmentation type every time call"""
    def __init__(self, tao=0.2, gamma=0.7, beta=0.2):
        self.data_augmentation_methods = [Crop(tao=tao), Mask(gamma=gamma), Reorder(beta=beta)]

    def __call__(self, sequence:List[int]) -> List[int]:
        # randint generate int x in range: a <= x <= b
        augment_method_idx = random.randint(0, len(self.data_augmentation_methods) - 1)
        augment_method = self.data_augmentation_methods[augment_method_idx]
        # print(augment_method.__class__.__name__) # debug usage
        return augment_method(sequence)

class Crop():
    """Randomly crop a subseq from the original sequence"""
    def __init__(self, tao=0.2):
        self.tao = tao

    def __call__(self, sequence: List[int]) -> List[int]:
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        sub_seq_length = int(self.tao*len(copied_sequence))
        #randint generate int x in range: a <= x <= b
        start_index = random.randint(0, len(copied_sequence)-sub_seq_length-1)
        if sub_seq_length<1:
            return [copied_sequence[start_index]]
        else:
            cropped_seq = copied_sequence[start_index:start_index+sub_seq_length]
            return cropped_seq

class Mask():
    """Randomly mask k items given a sequence"""
    def __init__(self, gamma=0.7):
        self.gamma = gamma

    def __call__(self, sequence:List[int]) -> List[int]:
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        mask_nums = int(self.gamma*len(copied_sequence))
        mask = [0 for i in range(mask_nums)]
        mask_idx = random.sample([i for i in range(len(copied_sequence))], k = mask_nums)
        for idx, mask_value in zip(mask_idx, mask):
            copied_sequence[idx] = mask_value
        return copied_sequence

class Reorder(object):
    """Randomly shuffle a continuous sub-sequence"""
    def __init__(self, beta=0.2):
        self.beta = beta

    def __call__(self, sequence:List[int]) -> List[int]:
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        sub_seq_length = int(self.beta*len(copied_sequence))
        start_index = random.randint(0, len(copied_sequence)-sub_seq_length-1)
        sub_seq = copied_sequence[start_index:start_index+sub_seq_length]
        random.shuffle(sub_seq)
        reordered_seq = copied_sequence[:start_index] + sub_seq + \
                        copied_sequence[start_index+sub_seq_length:]
        assert len(copied_sequence) == len(reordered_seq)
        return reordered_seq