from stable_baselines3 import DDPG, DQN, SAC, TD3  #, HerReplayBuffer
from her_replay_buffer_mod import HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.envs import BitFlippingEnv
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
import wandb
import her_wrappers

# wandb.init(project="gym-test")
conf = wandb.config
conf.learn_timesteps = 200
conf.n_sampled_goal = 4
conf.batch_size = 10
conf.max_episode_length = 50

env = her_wrappers.Robotics('FetchReach-v1')

# Available strategies (cf paper): future, final, episode
goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE

# If True the HER transitions will get sampled online
online_sampling = True
# Time limit for the episodes
max_episode_length = conf.max_episode_length

# Initialize the model
model = DDPG(
    "CnnPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=conf.n_sampled_goal,
        goal_selection_strategy=goal_selection_strategy,
        online_sampling=online_sampling,
        max_episode_length=max_episode_length
    ),
    verbose=1,
)

# Train the model
model.learn(conf.learn_timesteps)

model.save("./her_fetchreach")
# Because it needs access to `env.compute_reward()`
# HER must be loaded with the env
model = model_class.load('./her_fetchreach', env=env)

obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)

    if done:
        obs = env.reset()

