import os
import sys
import numpy as np
import torch
import pickle

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pprint import pprint
from munch import Munch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import cfg
from baseline_special.utils.utils import load_traces
from baseline_special.utils.constants import BITRATE_LEVELS
from plm_special.trainer import Trainer
from plm_special.trainer_u import Trainer as Trainer_u
from plm_special.evaluate import evaluate_on_env
from plm_special.test import test_on_env
from plm_special.data.dataset import ExperienceDataset
from plm_special.models.rl_policy import OfflineRLPolicy
from plm_special.models.state_encoder import EncoderNetwork
from plm_special.models.low_rank import peft_model, print_trainable_parameters
from plm_special.utils.utils import set_random_seed
from plm_special.utils.plm_utils import load_plm, load_mamba_plm
from plm_special.utils.console_logger import ConsoleLogger

from qcs.q_network import QNetwork
from qcs.utils import get_q_loss_mean
from baseline_special.utils.constants import S_INFO, S_LEN


def save_model(args, model, save_dir):
    mamba_model= model.mamba_plm
    save_path = os.path.join(save_dir, "mamba_model.bin")
    torch.save(mamba_model.state_dict(), save_path)
    torch.save(model.mamba_modules_except_plm.state_dict(), os.path.join(save_dir, 'modules_except_plm.bin'))

def load_plm_model(args, model, model_dir):
    if args.rank > 0:
        model.plm.load_adapter(model_dir, adapter_name='default')
        model.modules_except_plm.load_state_dict(torch.load(os.path.join(model_dir, 'modules_except_plm.bin'),weights_only=False))
    else:
        model.load_state_dict(torch.load(os.path.join(model_dir, 'model.bin')))
    return model

def load_mamba_model(args, model, model_dir):
    model.mamba_plm.load_state_dict(torch.load(os.path.join(model_dir, 'mamba_model.bin')))

    model.mamba_modules_except_plm.load_state_dict(torch.load(os.path.join(model_dir, 'modules_except_plm.bin'),weights_only=False))
    return model

def adapt(args, model, exp_dataset, exp_dataset_info, eval_env_settings, plm_model_dir,checkpoint_dir, best_model_dir, eval_process_reward_fn):
    if args.teacher:
        model = load_plm_model(args, model, plm_model_dir)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / args.warmup_steps, 1)
    )
    loss_fn = CrossEntropyLoss()
    if args.q:
        state_dim = S_INFO * S_LEN
        action_dim = 8
        hidden_dim = 256
        q_hiddens = 8
        layernorm = False

        model_path = args.q_path
        
        qf = QNetwork(state_dim, action_dim, hidden_dim, q_hiddens, layernorm).to('cuda')
        qf.load_state_dict(torch.load(model_path, map_location='cuda'))
        qf.eval()
        q_info=get_q_loss_mean(exp_dataset, qf=qf)

        trainer = Trainer_u(args, model=model, qf=qf, q_info=q_info, optimizer=optimizer,
                             exp_dataset=exp_dataset, loss_fn=loss_fn, device=args.device,
                               lr_scheduler=lr_scheduler, grad_accum_steps=args.grad_accum_steps)
    else:
        trainer = Trainer(args, model=model, optimizer=optimizer, exp_dataset=exp_dataset, loss_fn=loss_fn, device=args.device, lr_scheduler=lr_scheduler, 
                      grad_accum_steps=args.grad_accum_steps)

    target_return = exp_dataset_info.max_return * args.target_return_scale
    best_eval_return = 0.

    total_train_losses = []
    model_dir=os.path.dirname(checkpoint_dir)
    save_per_epoch = 5
    eval_per_epoch = 5
    for epoch in range(args.num_epochs):
        train_logs, train_losses = trainer.train_epoch()
        total_train_losses.extend(train_losses)
        print('='* 20, f'Training Iteration #{epoch}', '=' * 20)
        print('>' * 10, 'Training Information:')
        pprint(train_logs)
        
        if epoch % save_per_epoch == 0:  # save checkpoint
            checkpoint_dir_epoch = os.path.join(checkpoint_dir, str(epoch))
            if not os.path.exists(checkpoint_dir_epoch):
                os.makedirs(checkpoint_dir_epoch)
            save_model(args, model, checkpoint_dir_epoch)
            print('Checkpoint saved at:', checkpoint_dir_epoch)

        if epoch % eval_per_epoch == 0:
            eval_logs = evaluate_on_env(args, env_settings=eval_env_settings, model=model, target_return=target_return, max_ep_num=args.trace_num,
                                        process_reward_fn=eval_process_reward_fn)
            episodes_return = eval_logs['episodes_return']
            eval_reward= eval_logs['episodes_reward']
            if best_eval_return < episodes_return:
                best_eval_return = episodes_return
                save_model(args, model, best_model_dir)
                print('Best model saved at:', best_model_dir)

            eval_logs['best_return'] = best_eval_return
            print('>' * 10, 'Evaluation Information')
            pprint(eval_logs)
            with open(os.path.join(model_dir, 'eval_rewards.txt'), 'a') as f:
                f.write(f'Epoch {epoch}: Eval Reward: {eval_reward} Eval Return: {episodes_return}\n')

def test(args, model, exp_dataset_info, env_settings, model_dir, result_dir, test_process_reward_fn):
    model = load_mamba_model(args, model, model_dir)
    model.eval()

    print('Load model from:', model_dir)
    target_return = exp_dataset_info.max_return * args.target_return_scale
    results = test_on_env(args, model, result_dir, env_settings, target_return, args.trace_num, test_process_reward_fn, seed=args.seed)
    print(results)
    print('Test time:', results['time'], '\nlatences',results['latences'],'\nMean reward:', results['mean_reward'])
    print('Results saved at:', result_dir)

def run(args):
    assert args.plm_type in cfg.plm_types
    assert args.plm_size in cfg.plm_sizes
    assert args.exp_pool_path is not None, 'please specify a experience pool path for training'
    assert args.trace in cfg.trace_dirs.keys()
    assert args.video in cfg.video_size_dirs.keys()

    # 1. set seed
    set_random_seed(args.seed)

    # 2. create environment setting
    trace_dir = cfg.trace_dirs[args.trace]
    video_size_dir = cfg.video_size_dirs[args.video]
    all_cooked_time ,all_cooked_bw ,all_file_names, all_mahimahi_ptrs = load_traces(trace_dir)
    args.trace_num = min(args.trace_num, len(all_file_names))
    if args.trace_num == -1:
        args.trace_num = len(all_file_names)
    if args.trace_num == len(all_file_names):
        args.fixed_order = True

    env_settings = {
        'all_cooked_time': all_cooked_time,
        'all_cooked_bw': all_cooked_bw,
        'all_file_names': all_file_names,
        'all_mahimahi_ptrs': all_mahimahi_ptrs,
        'video_size_dir': video_size_dir,
        'fixed': args.fixed_order,
        'trace_num': args.trace_num,
        'training': args.adapt,
    }

    # 3. create training dataset, fetch info
    exp_pool = pickle.load(open(args.exp_pool_path, 'rb'))
    exp_dataset = ExperienceDataset(exp_pool, gamma=args.gamma, scale=args.scale, max_length=args.w, sample_step=args.sample_step)
    exp_dataset_info = Munch(exp_dataset.exp_dataset_info)
    print('Experience dataset info:')
    pprint(exp_dataset_info)
    
    # 4. create model
    
    # 4.1 load plm
    # args.device_out and args.device_mid are used for model parallelism (currently only support llama) 
    # For data/modules near the input side, we use args.device.
    # For data/modules near the output side, we use args.device_out.
    # For data/modules lying in the middle, we use args.device_mid (it can be None). 
    # If args.device == args.device_out == args.device_mid (if not None), everything will be the same as using only one device.
    if args.adapt and args.teacher:
        plm, *_ = load_plm(args.plm_type, cfg.plm_dir_llama, device_input_side=args.device, device_output_side=args.device_out, device_middle_side=args.device_mid)
        if args.rank != -1:
            plm = peft_model(plm, args.plm_type, rank=args.rank)
        #冻结教师模型 (Teacher 必须是冻结的)
        plm.eval()
        for param in plm.parameters():
            param.requires_grad = False
        print_trainable_parameters(plm)
    else:
        plm = None  # plm is not needed during testing, since we only use mamba plm for testing 
    mamba_plm, *_ = load_mamba_plm(device=args.device)
    mamba_plm=mamba_plm.to(args.device)
    
    # 4.2 create state encoder
    assert args.state_feature_dim is not None, 'please specify state feature dim to create state encoder'
    state_encoder = EncoderNetwork(embed_dim=args.state_feature_dim)
    state_encoder = state_encoder.to(args.device)

    # 4.3 create rl policy
    plm_embed_size = cfg.plm_embed_sizes[args.plm_type][args.plm_size]
    mamba_embed_size = cfg.mamba_config['mamba_embed_size']
    max_ep_len = exp_dataset_info.max_timestep + 1
    rl_policy = OfflineRLPolicy(state_feature_dim=args.state_feature_dim, bitrate_levels=BITRATE_LEVELS, 
                    state_encoder=state_encoder, plm=plm, mamba_plm=mamba_plm, plm_embed_size=plm_embed_size, mamba_embed_size=mamba_embed_size,
                    max_length=args.w, max_ep_len=max_ep_len, device=args.device, device_out=args.device_out, which_layer=args.which_layer)

    # 5. handling directory and path

    train_exp_pool_info = args.exp_pool_path.split('/')[-1][:-4]
    if args.q:
        plm_ft_dir = cfg.plm_ft_dir_q
    else:
        plm_ft_dir = cfg.plm_ft_dir
    models_dir = os.path.join(plm_ft_dir, train_exp_pool_info, \
                              f'rank_{args.rank}_w_{args.w}_epochs_{args.num_epochs}_seed_{args.seed}')
    results_dir = os.path.join(cfg.results_dir, f'{args.trace}', f'trace_num_{args.trace_num}_fixed_{args.fixed_order}',
                               f'rank_{args.rank}_w_{args.w}_tgt_scale_{args.target_return_scale}_seed_{args.seed}')
    results_dir = os.path.join(models_dir, 'test_results')
    checkpoint_dir = os.path.join(models_dir, 'checkpoint')
    best_model_dir = os.path.join(models_dir, 'best_model')

    # 6. start training/testing
    def process_reward(reward, 
                       max_reward=exp_dataset_info.max_reward, 
                       min_reward=exp_dataset_info.min_reward, 
                       scale=args.scale):
        reward = min(max_reward, max(min_reward, reward))  # bound reward
        return (reward - min_reward) / (max_reward - min_reward) / scale
    
    torch.backends.cudnn.benchmark = True

    if args.adapt:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)
        console_log = open(os.path.join(models_dir, 'console.log'), 'w')
        sys.stdout = ConsoleLogger(sys.__stdout__, console_log)
        plm_model_dir=args.plm_model_dir
        adapt(args, rl_policy, exp_dataset, exp_dataset_info, env_settings, plm_model_dir, checkpoint_dir, best_model_dir, process_reward)
    if args.test:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        model_dir = args.mamba_model_dir if args.mamba_model_dir is not None else best_model_dir
        assert os.path.exists(model_dir), f'Model weight dir {model_dir} does not exist.'
        test(args, rl_policy, exp_dataset_info, env_settings, model_dir, results_dir, process_reward)

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    # training dataset settings
    parser.add_argument('--exp-pool-path', help='', default='/home/ubuntu/kevin/netllm_qcs/artifacts/exp_pools/reward_make/reward_make.pkl')
    parser.add_argument('--sample-step', type=int, help='the steps for sampling experiences')
    # environment settings
    parser.add_argument('--trace', help='', type=str, default='valid')
    parser.add_argument('--trace-num', help=' ', type=int, default=-1)
    parser.add_argument('--video', help='', type=str, default='video1')
    parser.add_argument('--fixed-order', action='store_true', help='iterate over test traces in a fixed sequential order.')
    # plm settings
    parser.add_argument('--plm-type', type=str, default='llama')
    parser.add_argument('--plm-size', type=str, default='base')
    parser.add_argument('--rank', type=int, help='rank of low-rank matrices. if set to -1, low-rank matrices will not be enabled', default=128)
    # state encoder settings
    parser.add_argument('--state-feature-dim', type=int, help='feature dim of the state encoder', default=256)
    # rl policy related settings
    parser.add_argument('--w', type=int, help='context window for learning return distribution', default=10)
    parser.add_argument('--gamma', type=float, help='discounted factor of reward', default=1.)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-steps', type=int, default=2000)
    parser.add_argument('--num-epochs', type=int, default=161)
    parser.add_argument('--eval-per-epoch', type=int, help='', default=5)
    parser.add_argument('--save-checkpoint-per-epoch', type=int, help='',default=5)
    parser.add_argument('--target-return-scale', type=float, help='target return, which specifies the expected performance for the model to achieve', default=1.)
    parser.add_argument('--which-layer', type=int, help='for early stopping (not used in our experiments): specify which layer to stop (layer index starts from 0)', default=-1)
    # other settings
    parser.add_argument('--adapt', action="store_true", help='adapt model')
    parser.add_argument('--test', action="store_true", help='test model')
    parser.add_argument('--grad-accum-steps', dest='grad_accum_steps', type=int, default=32)
    parser.add_argument('--seed', help='random seed', type=int, default=100003)
    parser.add_argument('--scale', help='scale reward/return', type=int, default=1000)
    parser.add_argument('--device', action='store', dest='device', help='device (cuda or cpu) to run experiment')
    parser.add_argument('--device-out', action='store', dest='device_out', help='device (cuda or cpu) to place the split of model near the output')
    parser.add_argument('--device-mid', action='store', dest='device_mid', help='device (cuda or cpu) to place the split of model between the input and output')
    # for mamba4net
    parser.add_argument('--plm-model-dir', help='pretrained plm model dir for adaptation', default='')
    parser.add_argument('--alpha_kl', type=float, help='模型蒸馏 KL损失权重,默认1.0', default=1.0)
    parser.add_argument('--mamba-model-dir', help='', default='')
    #for qcs
    parser.add_argument('--q', action="store_true", help='use qcs to assist training')
    parser.add_argument('--q-path', type=str, help='path of the pretrained q network', \
                        default='/home/ubuntu/kevin/q_pretrain/exp/iql/q_train/q_train/112/qf_1000000.pth')
    args = parser.parse_args()

    if args.device_out is None:  
        args.device_out = args.device
    
    if args.save_checkpoint_per_epoch is None:
        args.save_checkpoint_per_epoch = args.eval_per_epoch
    assert args.save_checkpoint_per_epoch <= args.num_epochs

    print('Arguments:')
    pprint(args)

    run(args)
