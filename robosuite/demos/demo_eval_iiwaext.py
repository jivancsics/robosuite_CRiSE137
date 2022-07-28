"""
This script shows how to adapt an environment to be compatible
with the OpenAI Gym-style API. This is useful when using
learning pipelines that require supporting these APIs.

For instance, this can be used with OpenAI Baselines
(https://github.com/openai/baselines) to train agents
with RL.


We base this script off of some code snippets found
in the "Getting Started with Gym" section of the OpenAI
gym documentation.

The following snippet was used to demo basic functionality.

    import gym
    env = gym.make('CartPole-v0')
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

To adapt our APIs to be compatible with OpenAI Gym's style, this script
demonstrates how this can be easily achieved by using the GymWrapper.
"""

import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.controllers import load_controller_config

import gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        # Notice how the environment is wrapped by the wrapper
        env = GymWrapper(
            suite.make(
                env_id,
                robots="IIWA14_extended",  # use Sawyer robot
                use_camera_obs=False,  # do not use pixel observations
                has_offscreen_renderer=False,  # not needed since not using pixel obs
                has_renderer=False,  # make sure we can render to the screen
                reward_shaping=True,  # use dense rewards
                control_freq=20,  # control should happen fast enough so that simulation looks smooth
                horizon=1000,
                controller_configs=load_controller_config(default_controller="OSC_POSITION"),
            )
        )
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


if __name__ == "__main__":



    env_id = "Lift"
    env = GymWrapper(
        suite.make(
            env_id,
            robots="IIWA14_extended",  # use Sawyer robot
            use_camera_obs=False,  # do not use pixel observations
            has_offscreen_renderer=False,  # not needed since not using pixel obs
            has_renderer=True,  # make sure we can render to the screen
            reward_shaping=True,  # use dense rewards
            control_freq=20,  # control should happen fast enough so that simulation looks smooth
            horizon=1000,
            controller_configs=load_controller_config(default_controller="OSC_POSITION"),
        )
    )

    #model = SAC('MlpPolicy', eval_env)

    model_path = 'Logs_Lift/best_model.zip'
    model = SAC.load(model_path, env=env)
 



    for i_episode in range(20):
        obs = env.reset()
        print(obs)
        for t in range(500):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            #print(rewards)
            env.render()
            if dones:
                print("Episode finished after {} timesteps".format(t + 1))
                break
