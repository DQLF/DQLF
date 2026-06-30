import os

class Config:
    codes_path=os.path.dirname()

    _base_dir = os.getcwd()+'/'
    baseline_model_paths = {
        'pensieve': _base_dir + 'data/all_models/pensieve/nn_model_ep_79600.ckpt',
        'genet': _base_dir + 'data/all_models/genet/nn_model_ep_9800.ckpt',
    }
    
    trace_dirs = {
        'q_train': _base_dir + 'data/traces/q_train/',
        'test':codes_path + '/make_trace/select_trace/test/',
        'valid':_base_dir + 'data/traces/valid/',
        'train': _base_dir + 'data/traces/train/',
    }

    video_size_dirs = {
        'video1': _base_dir + 'data/videos/video1_sizes/',
        'video2': _base_dir + 'data/videos/video2_sizes/',
    }

    artifacts_dir = _base_dir + 'artifacts/'
    results_dir = artifacts_dir + 'results/'
    exp_pools_dir = artifacts_dir + 'exp_pools/'

    # plm special
    plm_types = [ 'llama']
    plm_sizes = [ 'small', 'base']  # note that the actual size of plm is dependent on the type of plm. 
                                                         # for example, for llama, 'base' is 7b, while for gpt2, 'base' is 340M. you can specify it yourself.
    plm_dir_llama =codes_path+'/llmModel/llama2_7B_hf/base'

    plm_q_ft_dir = _base_dir + 'data/ft_plms_u'
    plm_ft_dir = _base_dir + 'data/ft_plms'

    plm_embed_sizes = {
        'llama': {
            'base': 4096,
        },
    }
    plm_layer_sizes = {
        'llama': {
            'base': 32,
        },
    }
cfg = Config()
