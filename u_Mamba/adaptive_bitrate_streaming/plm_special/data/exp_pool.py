
class ExperiencePool:
    """
    Experience pool for collecting trajectories.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.traj_names = []
        self.traj_upbounds = []
        self.traj_rewards=[]

    def add(self, state, action, reward, done,traj_name,traj_upbound=None, traj_reward=None):
        self.states.append(state)  # sometime state is also called obs (observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.traj_names.append(traj_name)
        if traj_upbound is not None:
            self.traj_upbounds.append(traj_upbound)
        if traj_reward is not None:
            self.traj_rewards.append(traj_reward)

    def __len__(self):
        return len(self.states)

