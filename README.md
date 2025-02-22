## [K-ON: Knowledge On the Head Layer of Large Language Model](https://arxiv.org/pdf/2502.06257)
![](https://img.shields.io/badge/version-1.0.0-blue)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/zjukg/K-ON)
[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-orange)](https://huggingface.co/)
[![AAAI 2025](https://img.shields.io/badge/AAAI-2025-%23bd9f65?labelColor=%23bea066&color=%23ffffff)](https://aaai.org/conference/aaai/aaai-25/)

Welcome to the repository for K-ON. This project investigates the potential of LLMs in understanding and interacting with knowledge graphs, a domain that has received limited exploration in the context of NLP.

<div align="center">
    <img src="https://github.com/zjukg/K-ON/blob/master/imgs/example.png" width="55%" height="auto" />
</div>

### Overview

Recent advancements in large language models (LLMs) have significantly improved various natural language processing (NLP) tasks. Typically, LLMs are trained to predict the next token, aligning well with many NLP tasks. However, in knowledge graph (KG) scenarios, entities are the fundamental units and identifying an entity requires at least several tokens. This leads to a granularity mismatch between KGs and natural languages. To address this issue, we propose K-ON, which integrates KG knowledge into the LLM by employing multiple head layers for next k-step prediction. K-ON can not only generate entity-level results in one step, but also enables contrastive loss against entities, which is the most powerful tool in KG representation learning. Experimental results show that K-ON outperforms state-of-the-art methods that incorporate text and even the other modalities.


<div align="center">
    <img src="https://github.com/zjukg/K-ON/blob/master/imgs/arch.png" width="85%" height="auto" />
</div>


### Environment

To run the code, please first install all required packages:

```
pip install --upgrade pandas transformers peft==0.9 bitsandbytes swifter deepspeed easydict pyyaml
```


### Data Preprocessing

Then, we need to preprocess the datasets

```
python dataset.py -c config/DB15K.yaml
python dataset.py -c config/MKG-W.yaml
```


### Train and Evaluation

Run with the following scripts:

```
% for DBP15K dataset
sh scripts/run_db15k.sh
```

and

```
% for MKG-Y dataset
sh scripts/run_mkgy.sh
```

We use 8 GPUs to train K-ON, and you can modify this setting  ("gpu_ids" and "num_processes") in the above scripts. 

### Cite

Please condiser citing our paper if it is helpful to your work!

```bigquery
@inproceedings{K-ON,
  author       = {Lingbing Guo and
                  Yichi Zhang and
                  Zhongpu Bo and 
                  Zhuo Chen and 
                  Mengshu Sun and
                  Zhiqiang Zhang and
                  Yangyifei Luo and
                  Wen Zhang and
                  Huajun Chen},
  title        = {K-ON: Knowledge On the Head Layer of Large Language Model},
  booktitle    = {{AAAI}},
  year         = {2025}
}
```

### Thanks

We appreciate [LLaMA](https://github.com/facebookresearch/llama), [Huggingface Transformers](https://github.com/huggingface/transformers/tree/main/src/transformers/models/llama), [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html), [Alpaca-LoRA](https://github.com/tloen/alpaca-lora), and many other related works for their open-source contributions.

R.I.P. Kyoto Animation.

<div align="center">
    <img src="https://github.com/zjukg/K-ON/blob/master/imgs/banner.jpg" width="85%" height="auto" />
</div>

