"""
This file is rewritten based on openprompt.plms.__init__.py with a small modifications
on model class initialization.
We write this file just to avoid direct coding on openprompt source codes.
"""

import math
import torch
from typing import List, Optional
from collections import namedtuple
from yacs.config import CfgNode
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import LlamaConfig, LlamaTokenizer, LlamaTokenizerFast, \
     AutoConfig, AutoTokenizer, AutoModel,BitsAndBytesConfig

from plm_special.models.llama import LlamaModel

                    
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
    """
    Create device map for llama. The device map is used to evenly split the llama model into two/three parts on two devices.
    Currently only supoort llama-7b. We may consider to add support for more versions of llama.
    :param device_input_side: The device for the split of model that receives the input (e.g., 'cuda:0').
    :param device_output_side: The device for the split of model that produces the output (e.g., 'cuda:1'). 
    :param device_input_side: The device for the split of model that lies in the middle (e.g., 'cuda:2').
    :param device_map
    """
    num_layer=32
    device_map = {
        'embed_tokens': device_input_side
    }
    if device_middle_side is None:
        device_list = [device_input_side, device_output_side]
    else:
        device_list = [device_input_side, device_middle_side, device_output_side]
    for i in range(num_layer):  # llama-7b has 32 transformer blocks
        device_map[f'layers.{i}'] = device_list[i // math.ceil(num_layer / len(device_list))]
    device_map['norm'] = device_output_side
    return device_map
def create_device_map_for_mamba_ratio(
    device_input_side: str,
    device_output_side: str = None,
    device_middle_side: str = None
):
    num_layers = 64
    device_map = {
        'embed_tokens': device_input_side
    }

    # 如果只有一个设备
    if device_output_side is None:
        for i in range(num_layers):
            device_map[f"layers.{i}"] = device_input_side

        device_map["embeddings"] = device_input_side
        device_map["norm_f"] = device_input_side

    elif device_middle_side is None:
        num_first_device_layers = int(num_layers * 1 / 2)

        for i in range(num_layers):
            if i < num_first_device_layers:
                device_map[f"layers.{i}"] = device_input_side
            else:
                device_map[f"layers.{i}"] = device_output_side

        device_map["embeddings"] = device_input_side
        device_map["norm_f"] = device_output_side

    # 如果有三个设备
    else:
        device_list = [device_input_side, device_middle_side, device_output_side]
        layers_per_device = (num_layers + len(device_list) - 1) // len(device_list)

        for i in range(num_layers):
            device_idx = i // layers_per_device
            device_map[f"layers.{i}"] = device_list[device_idx]

        device_map["embeddings"] = device_input_side
        device_map["norm_f"] = device_output_side

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

        model = model_class.model.from_pretrained(
            model_path,
            config=model_config,
            device_map=device_map,
            # torch_dtype=torch.float16,
        )
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
    return model, tokenizer
