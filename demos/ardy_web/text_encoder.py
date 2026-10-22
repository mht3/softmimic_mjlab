"""Explicit pinned Llama + MNTP + supervised adapters for Transformers 5.

Transformers 5 no longer auto-loads a base model from an adapter-only directory
for the custom LlamaBiModel. Load the same published weights explicitly.
"""
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from peft import PeftModel
from ardy.model.llm2vec import LLM2Vec, LLM2VecEncoder
from ardy.model.llm2vec.models.bidirectional_llama import LlamaBiModel


def load_encoder(models):
    base=models['meta-llama/Meta-Llama-3-8B-Instruct']
    mntp=models['McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp']
    supervised=models['McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised']
    tokenizer=AutoTokenizer.from_pretrained(mntp,local_files_only=True)
    tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='left'
    model=LlamaBiModel.from_pretrained(base,dtype=torch.bfloat16,local_files_only=True)
    model=PeftModel.from_pretrained(model,mntp,local_files_only=True).merge_and_unload()
    model=PeftModel.from_pretrained(model,supervised,local_files_only=True)
    # This value controls ARDY's instruction formatting; a local snapshot path must not change it.
    model.config._name_or_path='meta-llama/Meta-Llama-3-8B-Instruct'
    config_path=Path(supervised)/'llm2vec_config.json'
    options=json.loads(config_path.read_text()) if config_path.exists() else {}
    encoder=LLM2VecEncoder.__new__(LLM2VecEncoder)
    encoder.llm_dim=4096
    encoder._device='cpu'
    encoder.model=LLM2Vec(model,tokenizer,**options).eval()
    for p in encoder.model.parameters(): p.requires_grad=False
    return encoder
