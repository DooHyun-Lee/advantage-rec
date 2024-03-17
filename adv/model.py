import torch
import torch.nn as nn
from typing import Set

class AdvFunc(nn.Module):
    def __init__(self, total_item: Set[int], args):
        super().__init__()
        self.args = args
        total_item_nums = len(total_item) + 2
        self.item_embs = nn.Embedding(total_item_nums, args.hidden_size,padding_idx=0)
        self.pos_embs = nn.Embedding(args.max_seq_len, args.hidden_size)

        self.layernorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.seq_encoder_layer = nn.TransformerEncoderLayer(d_model=args.hidden_size,
            nhead=args.num_attention_heads, dim_feedforward=args.hidden_size*4,
            dropout=args.attention_dropout_prob, activation='gelu',
            layer_norm_eps=1e-12, batch_first=True)
        self.seq_encoder = nn.TransformerEncoder(self.seq_encoder_layer,
            num_layers=args.num_hidden_layers)

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_vs(self, item_seq:torch.Tensor) -> torch.Tensor:
        '''
        return sequence embedding for each timestep 
        input : [b, max_len]
        output : [b, max_len, hidden_dim]
        '''
        # prepare mask for unidir transformer
        attn_mask = (item_seq > 0)
        attn_mask = attn_mask.unsqueeze(1)
        bsize, max_len = item_seq.size()
        nheads = self.args.num_attention_heads

        unidir_mask = torch.triu(torch.ones((1, max_len, max_len)), diagonal=1)
        unidir_mask = (unidir_mask ==0)
        attn_mask = attn_mask.to(item_seq.device)
        unidir_mask = unidir_mask.to(item_seq.device)
        attn_mask = attn_mask * unidir_mask
    
        attn_mask_final = attn_mask.unsqueeze(1).repeat(1, nheads, 1, 1)
        attn_mask_final = attn_mask_final.view(bsize*nheads, max_len, max_len)
        attn_mask_final = attn_mask_final.long()
        attn_mask_final = (1.0 - attn_mask_final) * -10000.0

        # prepare transformer input
        item_emb = self.item_embs(item_seq)
        pos_ids = torch.arange(max_len, dtype=torch.long, device=item_seq.device)
        pos_ids = pos_ids.unsqueeze(0).expand_as(item_seq)
        pos_emb = self.pos_embs(pos_ids)
        tot_emb = item_emb + pos_emb
        tot_emb = self.layernorm(tot_emb)
        tot_emb = self.dropout(tot_emb)

        seq_emb = self.seq_encoder(src=tot_emb, mask=attn_mask_final)
        return seq_emb

    def get_va(self, item_seq:torch.Tensor) -> torch.Tensor:
        '''
        return item embeddings  
        intput : [b, max_len]
        output : [b, max_len, hidden_dim]
        '''
        return self.item_embs(item_seq)

    def get_adv(self, item_seq:torch.Tensor, pos_item:torch.Tensor, neg_item:torch.Tensor) -> tuple[torch.Tensor]:
        '''
        return calculated advantage function values and corresponding target mask
        input : [b, max_len]
        output : [b*max_len], [b*max_len], [b*max_len]
        '''
        bsize, max_len = item_seq.size()
        vs = self.get_vs(item_seq)
        vs_flat = vs.view(-1, vs.size(2))

        va_pos = self.get_va(pos_item)
        va_neg = self.get_va(neg_item)
        va_pos_flat = va_pos.view(-1, va_pos.size(2))
        va_neg_flat = va_neg.view(-1, va_neg.size(2))

        adv_pos = torch.sum(vs_flat * va_pos_flat, -1)
        adv_neg = torch.sum(vs_flat * va_neg_flat, -1)
        target_mask = (pos_item>0).view(bsize * max_len).float()
        return adv_pos, adv_neg, target_mask

    def get_all_adv(self, item_seq:torch.Tensor):
        '''
        return advantage function values for all items 
        input : [b, max_len]
        output : [b, item_num]
        '''
        vs = self.get_vs(item_seq)[:, -1, :] # [b, hidden_dim]
        va = self.item_embs.weight # [item_num, hidden_dim]
        advs = torch.matmul(vs, va.T)
        return advs