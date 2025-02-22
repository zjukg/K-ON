import torch.nn as nn
import torch
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import re
from typing import Union
import os.path as osp
import pickle
import json
import swifter
from datasets import Dataset
import argparse, easydict, yaml

from typing import Any, Dict, List


class KONCollector:
    
    def __init__(self, dataset, tokenizer, num_neg=32):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.num_neg = num_neg
        self.all_ents = set(np.arange(self.dataset.num_ent).tolist())
    
    def strict_negative_sampling(self, triplet):
        filter_dict = self.dataset.train_filter_dict
        neg_ents = []
        
        # neg tails --- neg heads
        # 2 x batch size, num neg
        for (h,r,t) in triplet:
            a = list(self.all_ents-set(filter_dict[(h, r, -1)]))
            neg_ent = np.random.choice(a, size=self.num_neg, replace=False)
            neg_ents.append(neg_ent)
        for (h,r,t) in triplet:
            a = list(self.all_ents-set(filter_dict[(-1, r, t)]))
            neg_ent = np.random.choice(a, size=self.num_neg, replace=False)
            neg_ents.append(neg_ent)
        
        return np.stack(neg_ents)

    def filter_mask(self, triplet):
        filter_dict = self.dataset.test_filter_dict
        batch_size = len(triplet)
        fitler_mask_tail = np.ones([batch_size, self.dataset.num_ent])
        for fm, (h,r,t) in zip(fitler_mask_tail, triplet):
            pos = filter_dict[(h,r,-1)]
            fm[pos] = 0.
            fm[t] = 1.
        fitler_mask_head = np.ones([batch_size, self.dataset.num_ent])
        for fm, (h,r,t) in zip(fitler_mask_head, triplet):
            pos = filter_dict[(-1,r,t)]
            fm[pos] = 0.
            fm[h] = 1.
            
        return np.concatenate([fitler_mask_tail, fitler_mask_head])

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        first = features[0]
        batch_size = len(features)

        batch = {}
        for k, v in first.items():
            batch[k] = [f[k] for f in features]

        batch['input_text'] = batch['input_text']+batch['inv_input_text']
        self.tokenizer.add_eos_token=False
        batch.update(self.tokenizer(batch['input_text'], padding=True,))
        batch['input_length'] = np.sum(batch['attention_mask'], axis=1)

        split = batch['split'][0]
        if split=='train':
            neg_ents = self.strict_negative_sampling(np.stack([batch['h'], batch['r'], batch['t']], axis=1))
            batch['neg_ents'] = neg_ents
        else:
            filter_mask = self.filter_mask(np.stack([batch['h'], batch['r'], batch['t']], axis=1))
            batch['filter_mask'] = filter_mask
        
        # the real labels for prediction
        batch['pos_ents'] = batch['t'] + batch['h']
        
        
        del batch['input_text'], batch['inv_input_text'], batch['split']
        for k, v in batch.items():
            batch[k] = torch.tensor(batch[k])
            # print(k, batch[k].shape)
        batch['split'] = split

        if batch['split'] != 'train':
            # label means nothing, just for triggering the eval func
            return {'batch': easydict.EasyDict(batch), 'label': torch.ones(batch_size, dtype=torch.float)}
        else:
            return {'batch': easydict.EasyDict(batch)}



class Prompter(object):

    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


class KONDataset:
    def __init__(self, cfg=None):
        self.cfg = cfg
        base_dir = self.base_dir = cfg.base_dir
        self.prompter = Prompter('alpaca_short', verbose=False)
        
        
        
        # entities = pd.read_csv(base_dir+'entities.txt', sep=' ', header=None).values[:, 0]
        # relations = pd.read_csv(base_dir+'relations.txt', sep=' ', header=None).values[:, 0]
        train = pd.read_csv(base_dir+'train2id.txt', sep=' ', header=None, names=['h','t', 'r'])
        valid = pd.read_csv(base_dir+'valid2id.txt', sep=' ', header=None, names=['h','t', 'r'])
        test = pd.read_csv(base_dir+'test2id.txt', sep=' ', header=None, names=['h','t', 'r'])
        
        sep = ' ' if 'Kuai16K' not in base_dir else '\t'
        ent2text = pd.read_csv(base_dir+'entity2id.txt', sep=sep, header=None, names=['txt','id'])
        rel2text = pd.read_csv(base_dir+'relation2id.txt', sep=sep, header=None, names=['txt','id'])

        print('Refine entity and relation names...')
        if 'DB15K' in base_dir:
            ent2text['txt']=ent2text.txt.apply(lambda x: x.split('/')[-1][:-1].strip().replace('_',' '))
        elif 'MKG-W' in base_dir:
            ent2text['txt']=ent2text.txt.apply(lambda x: x.split('/')[-1].strip().replace('_',' '))
        elif 'MKG-Y' in base_dir:
            ent2text['txt']=ent2text.txt.apply(lambda x: x.strip().replace('_',' '))
        
        def refine_rel_txt(row):
            if 'DB15K' in base_dir:
                row = row.split('/')[-1][:-1].strip()
                if '#' in row:
                    row = row.split('#')[-1]
            row = re.sub("([a-z])([A-Z])","\g<1> \g<2>",row)
            return row.lower()
        
        rel2text['txt']=rel2text.txt.apply(refine_rel_txt)
        
        ent2text = pd.Series(ent2text['txt'].values, index=ent2text['id'].values)
        rel2text = pd.Series(rel2text['txt'].values, index=rel2text['id'].values)

        def get_er_vocab(data, er_vocab=None):
            if not er_vocab:
                er_vocab = defaultdict(list)
            for (h,r,t) in data[['h', 'r', 't']].values:
                er_vocab[(h, r, -1)].append(t)
                er_vocab[(-1, r, t)].append(h)
            return er_vocab
        train_filter_dict = get_er_vocab(train)
        test_filter_dict = get_er_vocab(pd.concat([train,valid,test],ignore_index=True))
        
        
        def add_name(data):
            data['hname'] = ent2text[data.h.values].values
            data['rname'] = rel2text[data.r.values].values
            data['tname'] = ent2text[data.t.values].values
        
        add_name(train)
        add_name(valid)
        add_name(test)
        
        def generate_text(row):
            hname, rname, tname = row['hname'], row['rname'], row['tname']
            instruction = 'Suppose that you are an expert in knowledge graphs, predict the tail entity of the given triplet (%s; %s; ?) with your knowledge. Please return only the name of the target entity' % (hname, rname)
            inv_instruction = 'Suppose that you are an expert in knowledge graphs, predict the head entity of the given triplet (?; %s; %s) with your knowledge. Please return only the name of the target entity' % (rname, tname)
            row['input_text'] = instruction
            row['inv_input_text'] = inv_instruction
            return row
        
        print('Generating instructional texts, please wait...')
        train = train.swifter.apply(generate_text, axis=1)
        valid = valid.swifter.apply(generate_text, axis=1)
        test = test.swifter.apply(generate_text, axis=1)
        
        train['split'] = 'train'
        valid['split'] = 'valid'
        test['split'] = 'test'
         
        self.train, self.valid, self.test, self.ent2text, self.rel2text = train, valid, test, ent2text, rel2text
        self.train_filter_dict = train_filter_dict
        self.test_filter_dict = test_filter_dict
        
        
        print('Convert to hf datasets...')
        self.train_data = Dataset.from_pandas(train)
        self.valid_data = Dataset.from_pandas(valid)
        self.test_data = Dataset.from_pandas(test)
        self.num_ent = self.ent2text.shape[0]
        self.num_rel = self.rel2text.shape[0]
        self.save()
        
    def save(self):
        saved_dir = self.base_dir
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)

        file_path = saved_dir+'preprocessed.pkl'
        print('##########Save dataset in %s############' % file_path)
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path):
        print('##########Load dataset from %s############' % file_path)
        with open(file_path, 'rb') as f:
            return pickle.load(f)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='data preprocessing')
    parser.add_argument("--config", "-c", type=str,
                        default='config/DB15K.yaml')
    parser.add_argument("--seed", "-s", type=str,
                        default=2042)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        cfg = easydict.EasyDict(yaml.safe_load(f))

    
    
    config_name = args.config.split('/')[-1].split('.')[0]
    args.config_name = config_name

    print('***************Read dataset from A*Net***************')
    print("Config file: %s" % args.config)
    print("Config name: %s" % args.config_name)
    print(cfg)
    
    dataset = KONDataset(cfg.dataset)
