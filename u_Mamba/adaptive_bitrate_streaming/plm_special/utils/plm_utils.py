
import math
from typing import List, Optional
from collections import namedtuple
from yacs.config import CfgNode
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import LlamaConfig, LlamaTokenizer, LlamaModel, LlamaTokenizerFast 

from plm_special.models.llama import LlamaModel
from plm_special.models.mamba_model.mamba_plms.mamba_llama import MambaLlamaNetworkingHeadModel
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer
from config import cfg 
# 假设类定义已导入
# from model_def import MambaLlamaNetworkingHeadModel 

import torch
import torch.nn as nn
from types import SimpleNamespace # 关键工具

def load_mamba_plm( device, **kwargs):
    # 1. 创建 Config 对象 (使用 SimpleNamespace 让它支持 .属性 访问)
    config = SimpleNamespace()
    
    # 复制 Llama 的关键参数 (确保 cfg 传进来了)
    config.hidden_size = cfg.mamba_config['hidden_size']
    config.mamba_embed_size = cfg.mamba_config['mamba_embed_size']
    config.num_hidden_layers = cfg.mamba_config['num_hidden_layers']
    
    # 设置 Mamba 参数 (带默认值)
    config.d_state = kwargs.get("d_state", 64)
    config.d_conv = kwargs.get("d_conv", 4)
    config.expand = kwargs.get("expand", 2)
    
    # 2. 实例化 Mamba 主干
    model = MambaLlamaNetworkingHeadModel(config)

    # 4. 设备处理
    model.to(device,dtype=torch.float32)
    
    print("Student Mamba Model initialized and ready")
    return model, config
                        
ModelClass = namedtuple("ModelClass", ('config', 'tokenizer', 'model'))

_MODEL_CLASSES = {
    "llama": ModelClass(**{
        "config": LlamaConfig,
        "tokenizer": LlamaTokenizer,
        "model": LlamaModel,
    }),

}

def get_model_class(plm_type: str):
    return _MODEL_CLASSES[plm_type]

def create_device_map_for_llama(device_input_side: str, device_output_side: str, device_middle_side: str=None):
    device_map = {
        'embed_tokens': device_input_side
    }
    if device_middle_side is None:
        device_list = [device_input_side, device_output_side]
    else:
        device_list = [device_input_side, device_middle_side, device_output_side]
    for i in range(32):  # llama-7b has 32 transformer blocks
        device_map[f'layers.{i}'] = device_list[i // math.ceil(32 / len(device_list))]
    device_map['norm'] = device_output_side
    return device_map

def load_plm(model_name, model_path, specials_to_add = None, **kwargs):
    model_class = get_model_class(plm_type = model_name)
    model_config = model_class.config.from_pretrained(model_path)

    if 'llama' in model_name:
        specials_to_add = ['<pad>']

    device_input_side = kwargs.pop('device_input_side', None)
    device_output_side = kwargs.pop('device_output_side', None)
    if 'llama' in model_name and device_input_side is not None and device_output_side is not None:
        device_middle_side = kwargs.pop('device_middle_side', None)
        device_map = create_device_map_for_llama(device_input_side, device_output_side, device_middle_side)
        model = model_class.model.from_pretrained(model_path, config=model_config, device_map=device_map)
    else:
        model = model_class.model.from_pretrained(model_path, config=model_config)
    
    tokenizer = model_class.tokenizer.from_pretrained(model_path) 
    model, tokenizer = add_special_tokens(model, tokenizer, specials_to_add=specials_to_add)
    
    return model, tokenizer, model_config

def add_special_tokens(model: PreTrainedModel,
                       tokenizer: PreTrainedTokenizer,
                       specials_to_add: Optional[List[str]] = None):
    if specials_to_add is None:
        return model, tokenizer
    for token in specials_to_add:
        if "pad" in token.lower():
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': token})
                model.resize_token_embeddings(len(tokenizer))
                print("pad token is None, set to id {}".format(tokenizer.pad_token_id))
    return model, tokenizer
