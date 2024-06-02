"""
This file implements a wrapper for facilitating compatibility with OpenAI gym (this is useful when using these
environments with code that assumes a gym-like interface like Garage) and compatibility with the special Meta RL
Learning structure in Garage. If a standard RL OpenAI gym wrapper for single task learning in Garage or comparable is
needed, set the single_task_ml flag properly at instantiation time.

"""

import numpy as np
from gym import spaces
from gym.core import Env
from garage._environment import EnvSpec

from robosuite.wrappers import Wrapper


class GymWrapper(Wrapper, Env):
    """
    Initializes the Gym wrapper for MRL with CRiSE Robosuite environments in Garage. Mimics many of the required
    functionalities of the Wrapper class found in the gym.core module. Handles the uniform representation of the
    observation vector over all MRL tasks. In case of single task MRL, the wrapper is able to adapt the
    observation vector through the single_task_ml flag.

    Args:
        env (MujocoEnv): The environment to wrap.
        single_task_ml (bool): Indicates whether to use the wrapper in a single task MRL setting (CRiSE 1/3) or
        in a general MRL setting with multiple diverse tasks (CRiSE 7).
        keys (None or list of str): If provided, each observation will
            consist of concatenated keys from the wrapped environment's
            observation dictionary. Defaults to proprio-state and object-state.

    Raises:
        AssertionError: [Object observations must be enabled if no keys]
    """

    def __init__(self, env, single_task_ml=False, keys=None):
        # Run super method
        super().__init__(env=env)
        # Create name for gym
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__

        # Get reward range
        self.reward_range = (0, self.env.reward_scale)

        # Init single_task_ml before using it in first _flatten_obs(obs) call
        self.single_task_ml = single_task_ml

        if keys is None:
            keys = []
            # Add object obs if requested
            if self.env.use_object_obs:
                keys += ["object-state"]
            # Add image obs if requested
            if self.env.use_camera_obs:
                keys += [f"{cam_name}_image" for cam_name in self.env.camera_names]
                if not self.single_task_ml:
                    raise Exception("USE OF CAMERA OBSERVATIONS IN THE OBSERVATION VECTOR "
                                    "-> META LEARNING ACROSS TASKS CURRENTLY NOT POSSIBLE!")
            # Iterate over all robots to add to state
            for idx in range(len(self.env.robots)):
                keys += ["robot{}_proprio-state".format(idx)]
        self.keys = keys

        # According to https://www.gymlibrary.dev/api/core/, self.metadata must be set to follow Gym's Environment
        # design rules. Due to the structure of Robosuite, the following render modes may be chosen.
        self.metadata = {'render_modes': [None, 'human']}

        # Set up observation and action spaces
        obs = self.env.reset()
        self.modality_dims = {key: obs[key].shape for key in self.keys}
        flat_ob = self._flatten_obs(obs)
        self.obs_dim = flat_ob.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low, high=high)

        # Gym specific attributes used in Garage
        self.max_path_length = self.env.horizon
        # adapt the env.spec to comply to the Garage structure by using class EnvSpec from garage/_environment.py
        self.spec = EnvSpec(observation_space=self.observation_space,
                            action_space=self.action_space,
                            max_episode_length=self.max_path_length)

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed

        Returns:
            np.array: observations flattened into a 1d array
        """
        env_name = str(self.env)
        ob_lst = []
        ob_lengths = []
        ob_lut = {}
        observation_dim = 33
        if 'IIWA14_extended_nolinear' in self.env.robot_names:
            robot_state_dim = 40
        else:
            robot_state_dim = 32

        if self.single_task_ml:
            ob_lst = []
            for key in self.keys:
                if key in obs_dict:
                    if verbose:
                        print("adding key: {}".format(key))
                    ob_lst.append(np.array(obs_dict[key]).flatten())
            return np.concatenate(ob_lst)

        else:
            for i, key in enumerate(self.keys):
                if key in obs_dict:
                    if verbose:
                        print("adding key: {}".format(key))
                    ob_lut[key] = i
                    ob_lst.append(np.array(obs_dict[key]).flatten())
                    ob_lengths.append(obs_dict[key].size)
                    if i == 1:
                        if ob_lengths[i] == robot_state_dim:
                            ob_lst = ob_lst[i:] + ob_lst[:i]

            if 'robosuite.environments.manipulation.door.Door' in env_name:
                ob_lst[1] = np.insert(ob_lst[1], [3, 3, 3, 3], [0, 0, 0, 0])    # door quat not known
                ob_lst[1] = np.insert(ob_lst[1], [7, 7, 7], ob_lst[1][10:13])  # door to end effector pos
                ob_lst[1] = np.insert(ob_lst[1], [10, 10, 10, 10], [0, 0, 0, 0])  # door to end effector quat not known
                ob_lst[1] = np.insert(ob_lst[1], [17, 17, 17, 17], [0, 0, 0, 0])  # handle quat not known
                ob_lst[1] = np.insert(ob_lst[1], [21, 21, 21], ob_lst[1][24:27])  # handle to end effector pos
                ob_lst[1] = np.insert(ob_lst[1], [24, 24, 24, 24], [0, 0, 0, 0])  # handle 2 end effector quat not known
                ob_lst[1] = np.delete(ob_lst[1], [28, 29, 30, 31, 32, 33])  # delete moved information
                ob_lst[1] = np.insert(ob_lst[1], [28, 28, 28], [0, 0, 0])  # Distance Cube2Cube info unknown

            elif 'robosuite.environments.manipulation.lift.Lift' in env_name:
                ob_lst[1] = np.insert(ob_lst[1], [10, 10, 10, 10], [0, 0, 0, 0])  # gripper 2 cube quat not known
                ob_lst[1] = np.insert(ob_lst[1], [14] * (observation_dim - 14),
                                      [0] * (observation_dim - 14))  # fill with zeros

            elif 'robosuite.environments.manipulation.nut_assembly.NutAssemblyRound' in env_name:
                ob_lst[1] = np.insert(ob_lst[1], [0] * 14,
                                      [0] * 14)  # fill square nut entries with zeros
                ob_lst[1] = np.insert(ob_lst[1], [28] * (observation_dim - 28),
                                      [0] * (observation_dim - 28))  # fill remaining entries with zeros

            elif 'robosuite.environments.manipulation.nut_assembly.NutAssemblySquare' in env_name:
                ob_lst[1] = np.insert(ob_lst[1], [14] * (observation_dim - 14),
                                      [0] * (observation_dim - 14))  # fill round nut entries with zeros

            elif 'robosuite.environments.manipulation.nut_assembly.NutAssembly' in env_name:
                ob_lst[1] = np.insert(ob_lst[1], [28] * (observation_dim - 28),
                                      [0] * (observation_dim - 28))  # fill with zeros

            elif 'robosuite.environments.manipulation.pick_place.PickPlaceMilk' in env_name:
                ob_lst[1] = np.insert(ob_lst[1], [14] * (observation_dim - 14),
                                      [0] * (observation_dim - 14))  # fill with zeros

            elif 'robosuite.environments.manipulation.pick_place.PickPlaceBread' in env_name:
                ob_lst[1] = np.insert(ob_lst[1], [14] * (observation_dim - 14),
                                      [0] * (observation_dim - 14))  # fill with zeros

            elif 'robosuite.environments.manipulation.pick_place.PickPlaceCereal' in env_name:
                ob_lst[1] = np.insert(ob_lst[1], [14] * (observation_dim - 14),
                                      [0] * (observation_dim - 14))  # fill with zeros

            elif 'robosuite.environments.manipulation.pick_place.PickPlaceCan' in env_name:
                ob_lst[1] = np.insert(ob_lst[1], [14] * (observation_dim - 14),
                                      [0] * (observation_dim - 14))  # fill with zeros

            elif 'robosuite.environments.manipulation.stack.Stack' in env_name:
                ob_lst[1] = np.insert(ob_lst[1], [7, 7, 7], ob_lst[1][14:17])  # cubeA 2 gripper pos
                ob_lst[1] = np.insert(ob_lst[1], [10, 10, 10, 10], [0, 0, 0, 0])  # cubeA 2 gripper quat not known
                ob_lst[1] = np.insert(ob_lst[1], [21, 21, 21], ob_lst[1][24:27])  # handle to end effector pos
                ob_lst[1] = np.insert(ob_lst[1], [24, 24, 24, 24], [0, 0, 0, 0])  # handle 2 end effector quat not known
                ob_lst[1] = np.delete(ob_lst[1], [28, 29, 30, 31, 32, 33])  # delete moved information
                ob_lst[1] = np.insert(ob_lst[1], [31, 31], [0, 0])  # door hinge and door handle qpos is zero

            else:
                raise Exception("No known Robosuite Meta learning environment found!")

            return np.concatenate(ob_lst)

    def reset(self):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict.

        Returns:
            np.array: Flattened environment observation space after reset occurs
        """
        ob_dict = self.env.reset()
        return self._flatten_obs(ob_dict)

    def step(self, action):
        """
        Extends vanilla step() function call to return flattened observation instead of normal OrderedDict.

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (np.array) flattened observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        ob_dict, reward, done, info = self.env.step(action)
        return self._flatten_obs(ob_dict), reward, done, info

    def seed(self, seed=None):
        """
        Utility function to set numpy seed

        Args:
            seed (None or int): If specified, numpy seed to set

        Raises:
            TypeError: [Seed must be integer]
        """
        # Seed the generator
        if seed is not None:
            try:
                np.random.seed(seed)
            except:
                TypeError("Seed must be an integer type!")

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward

        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]

        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.reward()
