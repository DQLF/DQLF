import os


class Config:
    _base_dir = '' if 'adaptive_bitrate_streaming' in os.getcwd() else 'adaptive_bitrate_streaming/'
    plm_model_dir={
        'genet':'/home/ubuntu/kevin/netllm_qcs/data/ft_plms/genet/llama_base/rank_128_w_10_ss_None_gamma_1.0_sfd_256_warm_2000_seed_100003_lr_0.0001_wd_0.0001/normalized_max_return_0.8_q_scale_4.0_epochs_161/best_model',
    }
    
    
    trace_dirs = {
        'q_train': _base_dir + 'data/traces/q_train/',
        'test':'/home/ubuntu/kevin/make_trace/select_trace/test/',
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
    plm_sizes = [ 'small', 'base', ]  # note that the actual size of plm is dependent on the type of plm. 
                                                         # for example, for llama, 'base' is 7b, while for gpt2, 'base' is 340M. you can specify it yourself.
    plm_dir_llama =''
    plm_ft_dir = _base_dir + 'data/ft_plms'
    plm_ft_dir_q = _base_dir + 'data/ft_plms_u'
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
    #for llama
    llama_config = dict(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048
    )

    #for mamba 
    mamba_config = dict(
        mamba_embed_size = 512,
        num_hidden_layers = 10,
        hidden_size = 512
    )
    

cfg = Config()
