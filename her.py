from stable_baselines3 import DDPG, DQN, SAC, TD3, HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.envs import BitFlippingEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import gym
import wandb
from wandb.keras import WandbCallback
import her_wrappers

# wandb.init(project="fetch-reach-her")
conf = wandb.config
conf.learn_timesteps = 50000
conf.n_sampled_goal = 4
conf.batch_size = 20
conf.max_episode_length = 50
conf.cnn = False
conf.reward_type='dense'

if conf.cnn:
    env = her_wrappers.Robotics('FetchReach-v1', reward_type=conf.reward_type)
else:
    env = gym.make('FetchReach-v1', reward_type=conf.reward_type)

# Available strategies (cf paper): future, final, episode
goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE

# If True the HER transitions will get sampled online
online_sampling = False
# Time limit for the episodes
max_episode_length = conf.max_episode_length

# Initialize the model
model = DDPG(
    "CnnPolicy" if conf.cnn else "MultiInputPolicy",
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

# Evaluate the model every 1000 steps
eval_callback = EvalCallback(eval_env=env, eval_freq=500, n_eval_episodes=10, log_path='logdir-her/dense/', verbose=True)

# Train the model
model.learn(conf.learn_timesteps, callback=eval_callback)

model.save("./her_fetchreach")
# Because it needs access to `env.compute_reward()`
# HER must be loaded with the env
model = DDPG.load('./her_fetchreach', env=env)

obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    print(reward)

    if done:
        obs = env.reset()

