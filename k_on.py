import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Union, OrderedDict
import torch
from torch import nn
import torch.nn.functional as F

from transformers import LlamaForCausalLM, LlamaConfig
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.models.llama.modeling_llama import LlamaRMSNorm

import peft

class KONAttn(nn.Module):
    def __init__(self, head, cfg):
        super().__init__()
        self.cfg = cfg
        self.head = head
        dtype = head.weight.data.dtype
        device = head.weight.device
        hidden_dim = head.weight.data.shape[-1]
        
        self.out_mlps = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=dtype, device=device), torch.nn.SiLU(), LlamaRMSNorm(hidden_dim)) for i in range(cfg.num_k_on)])
        self.k_on_attn = nn.MultiheadAttention(hidden_dim, cfg.num_attn_head, dropout=.3, batch_first=True, dtype=dtype, device=device)
        self.lora_heads = nn.ModuleList([peft.tuners.lora.Linear(head, 
                                                'K-ON-%i'%i, 
                                                r=cfg.r, 
                                                lora_alpha=cfg.lora_alpha, 
                                                lora_dropout=cfg.lora_dropout
                                                ) for i in range(cfg.num_k_on)])
        
        
    
    def forward(self, x):
        outputs = []
        for out_mlp in self.out_mlps:
            output = out_mlp(x) #+ x
            outputs.append(output)
        uncontioned = torch.stack(outputs, axis=1) # batch size, seq length, hidden dim
        attn_output, _ = self.k_on_attn(uncontioned, uncontioned, uncontioned, need_weights=False, is_causal=True) 
        attn_output =  attn_output*0.2 + x.unsqueeze(1)*0.5 + uncontioned*0.8  # attn with resnet attn_output +
        predictions = []
        for i in range(self.cfg.num_k_on):
            pred = self.lora_heads[i](attn_output[:, i])
            predictions.append(pred)
        predictions = torch.stack(predictions, axis=1)
        
        return predictions

class KONConfig(LlamaConfig):
    model_type = 'k_on_config'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class KON(LlamaForCausalLM):
    
    config_class = KONConfig

    def __init__(self, config):
        super().__init__(config)

    def init_kg_specs(self, ent2token, cfg,):
        self.ent2token = ent2token
        self.ent2tokenmask = ent2token==0
        self.num_ent = len(ent2token)
        self.kon_config = cfg
        self._init_k_on_head()
        
        self.sft_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.kld_loss = nn.KLDivLoss()

    

    def forward(
        self,
        batch,
        label=None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:

        input_ids, attention_mask, input_length = batch['input_ids'], batch['attention_mask'], batch['input_length']

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size = input_ids.shape[0]
        device = self.lm_head.weight.device
        
        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            # inputs_embeds=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        cache = transformer_outputs.past_key_values
        

        # batch_size, seq_len, hidden_state
        hidden_states = transformer_outputs[0]
        # select the last output of llm, batch_size x hidden_size
        logits = hidden_states[torch.arange(
            batch_size, device=hidden_states.device), input_length-1]
        
        preds = self.k_on(logits)
        
        pos_ents = batch['pos_ents']
        if self.training:
            # NCE loss
            neg_ents = batch['neg_ents']
            all_ents = torch.cat([pos_ents.unsqueeze(-1), neg_ents], axis=-1) # batch, 1 + num neg
            pred_all_ents, pred_token_prob = self.k_off(preds, all_ents, return_prob_seq=True)

            labels = torch.zeros_like(pred_all_ents)
            labels[:, 0] = 1.
            loss = self.loss(pred_all_ents, labels)
            
            htt_loss = self.tune(pos_ents, pred_token_prob, logits, cache, preds) * 1e-1
            loss += htt_loss
            
            return loss, pred_all_ents
        
        else:
            with torch.no_grad():
                pred_all_ents = self.k_off(preds) * batch['filter_mask'] # batch, num ent
                
                labels = torch.zeros_like(pred_all_ents)
                labels[range(batch_size), pos_ents] = 1.
                
                loss = self.loss(pred_all_ents, labels)
                
                pos_prob = pred_all_ents.gather(-1, pos_ents.unsqueeze(-1))
                ranking = torch.sum(pos_prob<=pred_all_ents, dim=-1) #+ 1
                return loss, ranking
    
    def tune(self, pos_ents, pred_token_prob, last_output, cache, k_on_pred):
        '''
        Brier's score
        https://stats.stackexchange.com/questions/46413/can-the-mean-squared-error-be-used-for-classification
        pos_ents: the positive (target) entities, batch size
        pred_token_prob: the one-shot prediction probabilities for all tokens of the positive entity, batch size x num k head
        cache: the LLM key and value cache for producing the original sequencial output probability for the positive entity
        '''
        batch_size = pos_ents.shape[0]
        input_ids = self.ent2token[pos_ents.cpu()].to(pred_token_prob.device) # batch size, token length
        attention_mask = ~self.ent2tokenmask[pos_ents.cpu()].to(pred_token_prob.device)
        

        transformer_outputs = self.model(
            input_ids=input_ids,
            # attention_mask=attention_mask,
            past_key_values=cache,
            # use_cache=True,
        )
        hidden_states = transformer_outputs[0] # batch size x token length x hidden size
        hidden_states = torch.cat([last_output.unsqueeze(1), hidden_states], axis=1) # # batch size x token length + 1 x hidden size
        
        pred = self.lm_head(hidden_states[:, :-1]) # batch size x token length x num token

        sft_loss = self.sft_loss(pred.flatten(end_dim=1), input_ids.flatten())
        kld_loss = self.kld_loss(k_on_pred.log_softmax(dim=-1), pred.detach().softmax(dim=-1))
        tune_loss = sft_loss + kld_loss
        return tune_loss
        
        
    
    def _init_k_on_head(self,):
        cfg = self.kon_config
        head = self.lm_head
        hidden_dim = head.weight.data.shape[-1]
        
        
        # self.k_on_weights = torch.ones(cfg.num_k_on, dtype=head.weight.data.dtype, device=head.weight.device, requires_grad=True)
        self.k_on_weights = torch.tensor([.95**i for i in range(cfg.num_k_on)], dtype=head.weight.data.dtype, device=head.weight.device, requires_grad=True)
        self.k_on_attn = KONAttn(head, cfg)
    
    def k_on(self, logits):
        '''
        logits: batch size x hidden
        '''
        return self.k_on_attn(logits)

    def k_off(self, preds, ents=None, return_prob_seq=False):
        '''
        preds: batch size, num k head, num token
        '''
        if ents is None:
            ents = torch.arange(self.num_ent).expand(preds.shape[0], self.num_ent) # batch size, num ent
        else:
            ents = ents.cpu()
        tokens = self.ent2token[ents].to(preds.device) # batch size, num ent, token length
        masks = self.ent2tokenmask[ents]
        lengths = (~masks).type(preds.dtype).sum(axis=-1).to(preds.device)

        
        
        probs = torch.gather(preds, 2, tokens.transpose(1,2)).transpose(1,2) # batch size, num ent, token_length
        probs[masks] = 0. # mask the paddings to 1.
        probs = probs * self.k_on_weights
        
        ent_probs = torch.sum(probs, axis=-1) / lengths # batch size, num ent
        if return_prob_seq:
            return ent_probs, probs[:, 0]
        else:
            return ent_probs
    
    def norm(self, prob):
        return torch.nn.functional.normalize(prob, p=2.0, dim=0)
    
    def loss(self, pred, target):
        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction="none")

        neg_weight = torch.ones_like(pred)
        if self.kon_config.adversarial_temperature > 0 and self.training:
            with torch.no_grad():
                neg_weight[:, 1:] = F.softmax(
                    pred[:, 1:] / self.kon_config.adversarial_temperature, dim=-1)
        else:
            neg_weight[:, 1:] = 1 / self.kon_config.num_neg
        loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
        
        loss = loss.mean()
        
        return loss
