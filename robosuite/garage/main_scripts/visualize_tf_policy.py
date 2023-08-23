# Load the policy
from garage.experiment import Snapshotter
import tensorflow as tf # optional, only for TensorFlow as we need a tf.Session
import robosuite as suite
from robosuite.wrappers import GymWrapper
import numpy as np
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from garage.tf.algos.rl2 import RL2Env
from garage.envs.gym_env import GymEnv
from time import sleep

if __name__ == "__main__":

    tf.compat.v1.disable_eager_execution()
    horizon = 500

    placement_initializer = UniformRandomSampler(
        name="ObjectSampler",
        x_range=[-0.1, -0.1],
        y_range=[0.1, 0.1],
        rotation=0,
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        reference_pos=np.array((0, 0, 0.8)),
        z_offset=0.01,
    )

    env = RL2Env(GymEnv(GymWrapper(suite.make(
        env_name="Lift",
        robots="IIWA14_extended_nolinear",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        placement_initializer=placement_initializer,
    ), single_task_ml=True)))

    snapshotter = Snapshotter()
    with tf.compat.v1.Session().as_default():  # optional, only for TensorFlow
        data = snapshotter.load('IIWA14/data/local/experiment/singleml_rl2_ppo_8')
        policy = data['algo'].policy

        # You can also access other components of the experiment
        # env = data['env']
        steps, max_steps = 0, 150
        done = False
        obs, _ = env.reset()  # The initial observation
        policy.reset()

        while steps < max_steps and not done:
            action, _ = policy.get_action(obs)
            envstep = env.step(action)
            obs = envstep.observation
            env.unwrapped.render()
            steps += 1
            sleep(0.02)

        env.close()