import torch
from torch.nn.utils.rnn import pad_sequence
import random
from typing import Tuple, List, Optional, Iterator, Dict, Callable, Union

def item_converter(items: List[int]) -> List[int]:
    return ['item_' + item for item in items]

def seq_generator(path:str) -> Tuple[List[List[int]]]:
    train_seq, valid_seq, test_seq = [], [], []
    with open(path, 'r') as f:
        for line in f:
            data = line.strip()
            data = data.split(' ')
            user= f'user_{data[0]}'
            items = item_converter(data[1:])
            for i in range(2, (len(items))):
                cur_list = ''.join(items[:i])
                prompt = f'Here is the purchase history list of {user}:[{cur_list}] try to recommend next item to the user' 
                temp_dict = dict()
                temp_dict['prompt'] = '\n\nHuman: ' + prompt
                temp_dict['target'] = '\n\nAssistant: ' + items[i]
                if i == len(items) -1:
                    test_seq.append(temp_dict)
                elif i == len(items) - 2:
                    valid_seq.append(temp_dict)
                else:
                    train_seq.append(temp_dict)
        return train_seq, valid_seq, test_seq

def get_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]: 
    def collate_fn(batch):
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                if 'prompt' in k:
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith('_input_ids'):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith('_labels'):
                    padding_value = -100
                elif k.endswith('_attention_mask'):
                    padding_value = 0
                
                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                if 'prompt' in k:
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]
        return padded_batch
    return collate_fn

def tokenize_batch_element(prompt:str, target:str, tokenizer, max_length:int) -> Dict:
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    target_tokens = tokenizer(target, add_special_tokens=False)

    assert tokenizer.eos_token_id not in prompt_tokens['input_ids']
    assert tokenizer.eos_token_id not in target_tokens['input_ids']

    target_tokens['input_ids'].append(tokenizer.eos_token_id)
    target_tokens['attention_mask'].append(1)

    assert len(prompt_tokens['input_ids']) + len(target_tokens['input_ids']) < max_length, f'current length : {len(prompt_tokens['input_ids']) + len(target_tokens['input_ids'])}'

    target_sequence_tokens = {k: prompt_tokens[k] + target_tokens[k] for k in target_tokens}
    target_sequence_tokens['labels'] = target_sequence_tokens['input_ids'][:]
    target_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])

    batch = {}
    batch['prompt'] = prompt
    batch['target'] = prompt + target
    batch['target_only'] = target

    for k, toks in {'prompt': prompt_tokens, 'target': target_sequence_tokens}.items():
        # input_ids, attn_mask, labels
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            batch[f'{k}_{type_key}'] = tokens
    return batch

def get_batch_iterator(tokenizer, 
                       seq: List[Dict],
                       batch_size: int=1,
                       shuffle: bool=True,
                       max_length: int=1024,
                       n_epochs: Optional[int]=None,
                       seed: int=42) -> Iterator[Dict]:
    random.seed(seed)
    flat_data = []
    for cur_seq in seq:
        flat_data.append((cur_seq['prompt'], cur_seq['target']))
    
    collate_fn = get_collate_fn(tokenizer)

    epoch_idx = 0
    while True:
        if epoch_idx >= n_epochs:
            break
        if shuffle:
            random.shuffle(flat_data)
        batch = []
        for prompt, target in flat_data:
            batch_element = tokenize_batch_element(prompt, target, tokenizer, max_length)
            batch.append(batch_element)
            if len(batch) == batch_size:
                yield collate_fn(batch)
                batch = []
        epoch_idx += 1