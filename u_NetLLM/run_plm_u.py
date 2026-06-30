import os
import sys
import numpy as np
import torch
import pickle
import yaml
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pprint import pprint
from munch import Munch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import cfg,codes_path,netllm_qcs_path
from baseline_special.utils.utils import load_traces
from baseline_special.utils.constants import BITRATE_LEVELS,S_INFO, S_LEN
from plm_special.trainer_u import Trainer_u
from plm_special.evaluate import evaluate_on_env
from plm_special.test import test_on_env
from plm_special.data.dataset import ExperienceDataset
from plm_special.models.rl_policy import OfflineRLPolicy
from plm_special.models.state_encoder import EncoderNetwork
from plm_special.models.low_rank import peft_model, verify_mamba_x_training
from plm_special.utils.utils import set_random_seed
from plm_special.utils.plm_utils import load_plm
from plm_special.utils.console_logger import ConsoleLogger

from qcs.q_network import QNetwork,ValueNetwork
from qcs.utils import get_q_loss_mean

def save_model(args, model, save_dir):
    if args.rank > 0:

        model.plm.save_pretrained(save_dir)
        torch.save(model.modules_except_plm.state_dict(), os.path.join(save_dir, 'modules_except_plm.bin'))
    else:
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.bin'))

def load_model(args, model, model_dir):
    if args.rank > 0:
        model.plm.load_adapter(model_dir, adapter_name='default', is_local=True)
        map_location = args.device_out 
        modules_except_plm_path = os.path.join(model_dir, 'modules_except_plm.bin')
        if not os.path.exists(modules_except_plm_path):
            raise FileNotFoundError(f"Checkpoint not found at {modules_except_plm_path}")
        state_dict = torch.load(modules_except_plm_path, map_location=map_location)
        model.modules_except_plm.load_state_dict(state_dict)
    else:
        model_path = os.path.join(model_dir, 'model.bin')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        map_location = args.device 
        state_dict = torch.load(model_path, map_location=map_location)
        model.load_state_dict(state_dict)
    return model
def load_qv_model(q_path,v_path):
    # ====== 加载 Q 网络 ======
        state_dim = S_INFO * S_LEN
        action_dim = 8
        hidden_dim = 256
        layernorm = False

        q_dp_path='/home/ubuntu/kevin/q_pretrain/exp/iql/q_train/q_train/113/qf_dp_1000000.pth'
        qf_dp = QNetwork(state_dim, action_dim, hidden_dim, 8, layernorm).to('cuda')
        qf_dp.load_state_dict(torch.load(q_dp_path, map_location='cuda'))
        qf_dp.eval()
        return qf_dp

def adapt(args, model, exp_dataset, exp_dataset_info, eval_env_settings, checkpoint_dir, best_model_dir,models_dir, eval_process_reward_fn):
    
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / args.warmup_steps, 1)
    )
    loss_fn = CrossEntropyLoss( reduction='none')

    target_return = exp_dataset_info.max_return * args.target_return_scale
    

    start_epoch = 0
    total_train_losses = []
    total_eval_returns = []
    best_eval_return = 0.
    best_eval_return_epoch = -1
    if args.resume:
        resume_path = os.path.join(checkpoint_dir, 'resume')
        resume_checkpoint_dir = os.path.join(resume_path, 'checkpoint')
        train_stats_path = os.path.join(resume_path,'train_stats.bin')
        if os.path.exists(train_stats_path):
            train_stats = torch.load(train_stats_path,map_location=args.device)
            # 加载模型参数
            model = load_model(args, model, resume_checkpoint_dir)
            optimizer.load_state_dict(train_stats['optimizer_state_dict'])

            lr_scheduler.load_state_dict(train_stats['scheduler_state_dict'])
            start_epoch = train_stats['epoch'] + 1
            total_train_losses = train_stats['train_losses']
            total_eval_returns = train_stats['eval_returns']
            best_eval_return = train_stats['best_eval_return']
            
            print(f'Checkpoint loaded from {resume_checkpoint_dir}\n starting from epoch: {start_epoch}\n best_eval_return: {best_eval_return}')
        else:
            raise(f'{train_stats_path} not found\n please check your resume path')
    else:
        print(f' start training from 0 epoch\n')
    exp_pool_name=args.exp_pool
    if args.q:
        name = f'{exp_pool_name}_q_{args.q}_normalized_max_return_{args.normalized_max_return}_q_scale_{args.q_scale}_epochs_{args.num_epochs}'
    else:
        name = f'{exp_pool_name}_q_{args.q}_epochs_{args.num_epochs}'
    
    if args.q:
        print('======== Using Q to assist training... ========')
        qf,qf_dp,vf=load_qv_model(args.q_path,args.v_path)
        q_info=get_q_loss_mean(exp_dataset, qf=qf)
        trainer = Trainer_u(args, model=model, qf=qf,qf_dp=qf_dp,vf=vf, q_info=q_info, optimizer=optimizer,
                             exp_dataset=exp_dataset, loss_fn=loss_fn, device=args.device,
                               lr_scheduler=lr_scheduler, grad_accum_steps=args.grad_accum_steps)
    
    for epoch in range(start_epoch,args.num_epochs):
        train_logs, train_losses = trainer.train_epoch()
        total_train_losses.extend(train_losses)
        
        print('='* 20, f'Training Iteration #{epoch}', '=' * 20)
        print('>' * 10, 'Training Information:')
        pprint(train_logs)
        save_per_epoch = args.save_checkpoint_per_epoch
        eval_per_epoch = args.eval_per_epoch
        save_train_stats_per_epoch = 10
        if epoch < 60:
            save_per_epoch = 10
            eval_per_epoch = 10
        else:
            save_per_epoch = 5
            eval_per_epoch = 5
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
            episodes_len = eval_logs['episodes_len']
            episodes_reward = eval_logs['episodes_reward']

            total_eval_returns.append((epoch,episodes_return,episodes_reward))
            if best_eval_return < episodes_return:
                best_eval_return = episodes_return
                best_eval_return_epoch = epoch
                save_model(args, model, best_model_dir)
                print('New best model with return {:.6f} at epoch {}'.format(best_eval_return, best_eval_return_epoch))
                print('Best model saved at:', best_model_dir)

            eval_logs['best_return'] = best_eval_return
            eval_logs['best_return_epoch'] = best_eval_return_epoch
            print('>' * 10, 'Evaluation Information')
            pprint(eval_logs)
            best_eval_returns_path = os.path.join(models_dir, 'best_eval_return')
            np.savetxt(best_eval_returns_path, list([best_eval_return_epoch,best_eval_return,episodes_len]), fmt='%.6f', delimiter='\n')
        if epoch % save_train_stats_per_epoch == 0:
            print('Saving train stats...')
            resume_path = os.path.join(checkpoint_dir, 'resume')
            if not os.path.exists(resume_path):
                os.makedirs(resume_path)
            resume_checkpoint_dir = os.path.join(resume_path, 'checkpoint')
            if not os.path.exists(resume_checkpoint_dir):
                os.makedirs(resume_checkpoint_dir)
            save_model(args, model, resume_checkpoint_dir)
            train_stats={
                'model_state_dict':model.state_dict() if args.rank <= 0 else None,
                'modules_except_plm_state_dict':model.modules_except_plm.state_dict() if args.rank > 0 else None,
                'optimizer_state_dict':optimizer.state_dict(),
                'scheduler_state_dict':lr_scheduler.state_dict(),
                'train_losses':total_train_losses,
                'eval_returns':total_eval_returns,
                'best_eval_return':best_eval_return,
                'epoch':epoch
            }
            torch.save(train_stats, os.path.join(resume_path,'train_stats.bin'))
            print('Train stats saved at:', resume_path)
        # save training losses , eval_return, plot
        train_losses_path = os.path.join(models_dir, 'train_losses.txt')
        eval_returns_path = os.path.join(models_dir, 'eval_returns.txt')
        
        np.savetxt(train_losses_path, total_train_losses, fmt='%.6f', delimiter='\n')
        np.savetxt(
            eval_returns_path,
            np.array(total_eval_returns, dtype=np.float64),  # shape=(N,3)
            fmt='%d\t%.6f\t%.6f',
            header='epoch\treturn\treward',
            comments=''
        )

def test(args, model, exp_dataset_info, env_settings, model_dir, result_dir, models_dir,test_process_reward_fn):

    model = load_model(args, model, model_dir)
    print('Load model from:', model_dir)
    target_return = exp_dataset_info.max_return * args.target_return_scale
    lambda_q=args.lambda_q
    if args.test_with_q:
        print('======== Using Q/V to assist testing... ========')
        qf,vf=load_qv_model(args.q_path,args.v_path)
        
    else:
        qf=None
        vf=None
        print('======== Not using Q/V ========')
    results = test_on_env(args, model, result_dir, env_settings, target_return, args.trace_num, test_process_reward_fn, qf=qf,vf=vf,lambda_q=lambda_q,seed=args.seed)

    print('Test time:', results['time'],'\nlatences:', results['latences'], '\nMean reward:', results['mean_reward'], '\nMean reward all:', results['mean_reward_all'])
    best_eval_returns_path = os.path.join(models_dir, 'best_eval_return')
    with open(best_eval_returns_path, 'r') as f:
        best_eval_return = f.readlines()
    best_return = best_eval_return[1]
    #save bestreturn和testresult
    test_result_path = os.path.join(models_dir, f'{args.trace}_eval_{float(best_return):.6f}_test_{results["mean_reward"]:.6f}')
    with open(test_result_path, 'w', encoding='utf-8') as f:
        f.write(f'best_eval_return:{best_return}\n')
        f.write(str(results))


def run(args):
    assert args.plm_type in cfg.plm_types
    assert args.plm_size in cfg.plm_sizes
    assert args.exp_pool is not None, 'please specify a experience pool path for training'
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
        'training': args.adapt
    }

    # 3. create training dataset, fetch info
    exp_pool_path=cfg.exp_pool[args.exp_pool]
    exp_pool = pickle.load(open(exp_pool_path, 'rb'))
    exp_dataset = ExperienceDataset(exp_pool, gamma=args.gamma, scale=args.scale, max_length=args.w, sample_step=args.sample_step)
    exp_dataset_info = Munch(exp_dataset.exp_dataset_info)
    print('Experience dataset info:')
    print(len(exp_pool)/47)
    pprint(exp_dataset_info)
    
    # 4. create model
    
    # 4.1 load plm
    # args.device_out and args.device_mid are used for model parallelism (currently only support llama) 
    # For data/modules near the input side, we use args.device.
    # For data/modules near the output side, we use args.device_out.
    # For data/modules lying in the middle, we use args.device_mid (it can be None). 
    # If args.device == args.device_out == args.device_mid (if not None), everything will be the same as using only one device.
    if args.plm_type == 'llama':
        model_path = cfg.plm_dir_llama
    else:
        model_path = args.plm_path
    plm, *_ = load_plm(args.plm_type, model_path, 
                       device_input_side=args.device, device_output_side=args.device_out, device_middle_side=args.device_mid)
    if args.rank != -1:
        plm = peft_model(plm, args.plm_type, rank=args.rank,modules=args.peft_modules)

    # 4.2 create state encoder
    assert args.state_feature_dim is not None, 'please specify state feature dim to create state encoder'
    state_encoder = EncoderNetwork(embed_dim=args.state_feature_dim)
    state_encoder = state_encoder.to(args.device)

    # 4.3 create rl policy
    plm_embed_size = cfg.plm_embed_sizes[args.plm_type][args.plm_size]
    max_ep_len = exp_dataset_info.max_timestep + 1
    rl_policy = OfflineRLPolicy(state_feature_dim=args.state_feature_dim, bitrate_levels=BITRATE_LEVELS, state_encoder=state_encoder, plm=plm, plm_embed_size=plm_embed_size, 
                                           max_length=args.w, max_ep_len=max_ep_len, device=args.device, device_out=args.device_out, which_layer=args.which_layer)

    # 5. handling directory and path

    # extract training experience pool information
    train_exp_pool_info = args.exp_pool

    modules = ''
    for module in args.peft_modules:
        modules += '_'+module
    plm_ft_dir = cfg.plm_q_ft_dir if args.q  else cfg.plm_ft_dir
    models_dir = os.path.join(plm_ft_dir, train_exp_pool_info,f'{args.plm_type}_{args.plm_size}', f'rank_{args.rank}_w_{args.w}_ss_{args.sample_step}_gamma_{args.gamma}_sfd_{args.state_feature_dim}_warm_{args.warmup_steps}_seed_{args.seed}_lr_{args.lr}_wd_{args.weight_decay}',\
                    f'normalized_max_return_{args.normalized_max_return}_q_scale_{args.q_scale}_epochs_{args.num_epochs}')
    results_dir = os.path.join(models_dir, 'test_results',f'{args.trace}_{args.fixed_order}')
    checkpoint_dir = os.path.join(models_dir, f'checkpoint')
    best_model_dir = os.path.join(models_dir, f'best_model')
    # 保存args
    os.makedirs(models_dir, exist_ok=True)
    args_dict = vars(args)
    with open(models_dir+"/args_config.yaml", "w") as f:
        yaml.dump(args_dict, f)

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
        console_log = open(os.path.join(models_dir, f'console.log'), 'w')
        sys.stdout = ConsoleLogger(sys.__stdout__, console_log)
        adapt(args, rl_policy, exp_dataset, exp_dataset_info, env_settings, checkpoint_dir, best_model_dir, models_dir ,process_reward)
    if args.test:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        model_dir = args.model_dir if args.model_dir is not None else best_model_dir
        if args.model_dir is not None :
            if model_dir.split('/')[-1] == 'best_model':
                models_dir = '/'.join(args.model_dir.split('/')[:-1])
            else:
                models_dir = '/'.join(args.model_dir.split('/')[:-2])
            results_dir = os.path.join(models_dir, 'test_results',f'{args.trace}_{args.fixed_order}')
            os.makedirs(results_dir, exist_ok=True)
        assert os.path.exists(model_dir), f'Model weight dir {model_dir} does not exist.'
        test(args, rl_policy, exp_dataset_info, env_settings, model_dir, results_dir,models_dir, process_reward)



if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    # training dataset settings
    parser.add_argument('--exp-pool', type=str,default='genet')
    parser.add_argument('--sample-step', type=int, help='the steps for sampling experiences')
    # environment settings
    
    parser.add_argument('--trace', help='', type=str, default='train')
    parser.add_argument('--trace-num', help='number of traces. if set to -1, use all traces in the trace dir.', type=int, default=-1)
    parser.add_argument('--video', help='name of video (e.g., video1)', type=str, default='video1')
    parser.add_argument('--fixed-order', action='store_true', help='iterate over test traces in a fixed sequential order.')
    # plm settings
    parser.add_argument('--plm-type', type=str, default='llama')
    parser.add_argument('--plm-size', type=str, default='base')
    parser.add_argument('--rank', type=int, help='rank of low-rank matrices. if set to -1, low-rank matrices will not be enabled', default=128)
    parser.add_argument('--peft-modules', nargs='+', help ='', required=True)
    # state encoder settings
    parser.add_argument('--state-feature-dim', type=int, help='feature dim of the state encoder', default=256)
    # rl policy related settings
    parser.add_argument('--w', type=int, help='context window for learning return distribution', default=10)
    parser.add_argument('--gamma', type=float, help='discounted factor of reward', default=1.)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-steps', type=int, default=2000)
    parser.add_argument('--num-epochs', type=int, default=160)
    parser.add_argument('--eval-per-epoch', type=int, help='evaluation per epoch', default=2)
    parser.add_argument('--save-checkpoint-per-epoch', type=int, help='saving checkpoint per iteration',default=2)
    parser.add_argument('--target-return-scale', type=float, help='target return, which specifies the expected performance for the model to achieve', default=1.)
    parser.add_argument('--which-layer', type=int, help='for early stopping (not used in our experiments): specify which layer to stop (layer index starts from 0)', default=-1)
    # other settings
    parser.add_argument('--adapt', action="store_true", help='adapt model')
    parser.add_argument('--test', action="store_true", help='test model')
    parser.add_argument('--resume', type=int, help='resume training',default=0)
    parser.add_argument('--grad-accum-steps', dest='grad_accum_steps', type=int, default=32)
    parser.add_argument('--seed', help='random seed', type=int, default=100003)
    parser.add_argument('--scale', help='scale reward/return', type=int, default=1000)
    parser.add_argument('--model-dir', help='model weight dir for testing',\
                        default=None )
    parser.add_argument('--device', action='store', dest='device', help='device (cuda or cpu) to run experiment')
    parser.add_argument('--device-out', action='store', dest='device_out', help='device (cuda or cpu) to place the split of model near the output')
    parser.add_argument('--device-mid', action='store', dest='device_mid', help='device (cuda or cpu) to place the split of model between the input and output')
    #for qcs
    parser.add_argument('--q', action="store_true", help='use Q to assist training')
    parser.add_argument('--q-path', type=str, help='path of the pretrained q network', \
                        default=cfg.q_path)
    parser.add_argument('--normalized_max_return', type=float, help='for calculate q_alpha in weighted_q_loss',\
                         default=0.8)
    parser.add_argument('--q-scale', type=float, help='scale for q value', default=4.0)
    parser.add_argument('--lambda-q', type=float, help='lambda for q value in testing', default=0.1)
    parser.add_argument('--swanlab', action="store_true", help='use swanlab to log experiment')
    
    args = parser.parse_args()


    if args.device_out is None:  
        args.device_out = args.device
    
    if args.save_checkpoint_per_epoch is None:
        args.save_checkpoint_per_epoch = args.eval_per_epoch
    assert args.save_checkpoint_per_epoch <= args.num_epochs

    print('Arguments:')
    pprint(args)

    run(args)
