import os
import numpy as np
import torch
import argparse
import os
from tqdm import trange
from coolname import generate_slug
import swanlab
import utils
from models.iql.base import IQL
from plm_special.data.exp_pool import ExperiencePool
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--seed", default=112, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--save_freq", default=2e5, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=4e6, type=int)   # Max time steps to run environment
    parser.add_argument("--save_model", action="store_true", default=True)        # Save model and optimizer parameters
    parser.add_argument("--log_to_wb", "-w", type=bool, default=False)
    parser.add_argument("--normalize", default=False, action='store_true')
    # IQL
    parser.add_argument("--lr", default=3e-4, type=float)      # Max episode length
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--expectile", default=0.7, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
    parser.add_argument("--q_hiddens", default=2, type=int)
    parser.add_argument("--v_hiddens", default=2, type=int)
    parser.add_argument("--layernorm", default=False, action='store_true')
    parser.add_argument("--save_dir", type=str, default="./exp")
    parser.add_argument("--exp-pool",type=str,default="exp_pool.pkl")
    args = parser.parse_args()
    args.cooldir = generate_slug(2)

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")


    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = 48
    action_dim = 8
    max_action = 7

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    # replay_buffer_use_rtg = utils.ReplayBuffer(state_dim, action_dim)
    exp_pool_name = args.exp_pool
    exp_pool = pickle.load(open(f'plm_special/data/{exp_pool_name}.pkl', 'rb'))

    replay_buffer.convert_abr(exp_pool)
    # replay_buffer_use_rtg.convert_abr_use_reward(exp_pool)

    if args.normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean, std = 0, 1
        
    if args.log_to_wb:
        swanlab.init(
            name=f'exp_pool_{args.exp_pool}_use_dp',
            project="abr-Q-Pretrain",
            config=args
        )

    # Build work dir
    utils.mkdir(args.save_dir)
    base_dir = os.path.join(args.save_dir, f"{(args.policy).lower()}")
    utils.mkdir(base_dir)
    args.work_dir = os.path.join(base_dir, 'q_train')
    utils.mkdir(args.work_dir)
    args.work_dir = os.path.join(args.work_dir, args.exp_pool)
    utils.mkdir(args.work_dir)
    args.model_dir = os.path.join(args.work_dir, str(args.seed))
    utils.mkdir(args.model_dir)
        
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        # IQL
        "discount": args.discount,
        "tau": args.tau,
        "expectile": args.expectile,
        "hidden_dim": args.hidden_dim,
        "q_hiddens": args.q_hiddens,
        "v_hiddens": args.v_hiddens,
        "layernorm": args.layernorm,
    }

    # Initialize policy
    policy = IQL(**kwargs)

    for t in trange(int(args.max_timesteps)):
        policy.train_use_dp(replay_buffer, args.batch_size, log_to_wb=args.log_to_wb)
        
        if (t + 1) % args.save_freq == 0:
            policy.save(args.model_dir)
    policy.save(args.model_dir)