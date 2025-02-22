import os, sys, logging, argparse, yaml, easydict

import numpy as np
import torch

from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer
)
from transformers.trainer import Trainer
from peft import (
    LoraConfig,
    get_peft_model,
)
from accelerate import Accelerator

from k_on import *
from dataset import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='data preprocessing')
    parser.add_argument("--config", "-c", type=str,
                        default='config/DB15K.yaml')
    parser.add_argument("--seed", "-s", type=int,
                        default=2042)
    parser.add_argument("--num_neg", "-n", type=int,
                        default=-1) 
    parser.add_argument("--num_k_on", "-k", type=int,
                        default=-1)                                        
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = easydict.EasyDict(yaml.safe_load(f))

    if args.num_neg>0:
        cfg.k_on.num_neg = args.num_neg
    if args.num_k_on>0:
        cfg.k_on.num_k_on = args.num_k_on

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = KONDataset.load('%spreprocessed.pkl'%cfg.dataset.base_dir)

    config_name = args.config.split('/')[-1].split('.')[0]


    args.config_name = config_name
    cfg.trainer.output_dir += config_name
    
    print("Config file: %s" % args.config)
    print(cfg)
    
    
    print('***************Load tokenizer***************')
    tokenizer = AutoTokenizer.from_pretrained(**cfg.tokenizer)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'right'
    tokenizer.add_bos_token=False
    tokenizer.add_eos_token=False
    ent2token = tokenizer(dataset.ent2text.values.tolist(), truncation=True, padding=True, max_length=cfg.k_on.num_k_on).input_ids
    ent2token = torch.tensor(ent2token)
    dataset.ent2token = ent2token
    tokenizer.add_bos_token=True
    # tokenizer.add_eos_token=False

    config = KONConfig.from_pretrained(**cfg.llm_config)
    model = KON.from_pretrained(
        **cfg.llm, device_map={"": Accelerator().process_index}, config=config)

    lora_config = LoraConfig(**cfg.loraconfig)
    model = get_peft_model(model, lora_config)

    
    model.init_kg_specs(ent2token, cfg.k_on) 
    

    print(model.print_trainable_parameters())
    print(model)

    

    data_loader = KONCollector(dataset, tokenizer, cfg.k_on.num_neg)
    
    training_args = TrainingArguments(**cfg.trainer)
    print(training_args)


    def compute_metrics(predictions):
        ranking = predictions[0].astype(float)
        metric = ("mr", "mrr", "hits@1", "hits@3", "hits@10")
        results = {}
        for _metric in metric:
            if _metric == "mr":
                score = ranking.mean()
            elif _metric == "mrr":
                score = (1 / ranking).mean()
            elif _metric.startswith("hits@"):
                threshold = int(_metric[5:])
                score = (ranking <= threshold).mean()
            else:
                raise ValueError("Unknown metric `%s`" % _metric)

            results[_metric] = score
        print(results)
        return results

    removed_columns = ['hname','rname','tname']

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset.test_data.remove_columns(
            removed_columns),  
        train_dataset=dataset.train_data.remove_columns(removed_columns),
        data_collator=data_loader,
        compute_metrics=compute_metrics
    )
    trainer.evaluate()
    trainer.train()

