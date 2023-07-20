import robosuite as suite
from robosuite.wrappers import GymWrapper
import pickle


# from garage.envs import GymEnv


class SawyerWipeRobosuiteEnv:
    """
    This class encapsulates the wipe task of Robosuite. The behaviour of MetaWorld environments after instantiation
    via metaworld.ML10() shall be copied by this class.

    Class variables (inherited from args in robosuite/environments/manipulation/wipe.py):

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory.
            For this environment, setting a value other than the default ("WipingGripper") will raise an
            AssertionError, as this environment is not meant to be used with any other alternative gripper.

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

        task_config (None or dict): Specifies the parameters relevant to this task. For a full list of expected
            parameters, see the default config dict at the top of the file robosuite/environments/manipulation/wipe.py.
            If None is specified, the default configuration will be used.
    """

    def __init__(self):
        self.env_configuration = "default"
        self.controller_configs = None
        self.gripper_types = "WipingGripper"
        self.initialization_noise = "default"
        self.use_camera_obs = False
        self.use_object_obs = True
        self.reward_scale = 1.0
        self.reward_shaping = True
        self.has_renderer = False
        self.has_offscreen_renderer = False
        self.render_camera = "frontview"
        self.render_collision_mesh = False
        self.render_visual_mesh = True
        self.render_gpu_device_id = -1
        self.control_freq = 20
        self.horizon = 500
        self.ignore_done = False
        self.hard_reset = True
        self.camera_names = "agentview"
        self.camera_heights = 256
        self.camera_widths = 256
        self.camera_depths = False
        self.camera_segmentations = None
        self.task_config = None
        self.renderer = "mujoco"
        self.renderer_config = None

        # Necessary for setting the subtasks correctly
        self._set_task_called = False
        self._freeze_rand_vec = True
        self._last_rand_vec = None

    def set_task(self, task):
        self._set_task_called = True
        data = pickle.loads(task.data)
        del data["env_cls"]
        self._last_rand_vec = data["rand_vec"]
        self._freeze_rand_vec = True
        self._last_rand_vec = data["rand_vec"]
        del data["rand_vec"]
        # initialize self.placement_initializer correctly here!!!
        # self.placement_initializer(self._last_rand_vec)

    def __call__(self):
        return GymWrapper(
            suite.make(
                "Wipe",
                robots="Sawyer",
                env_configuration=self.env_configuration,
                controller_configs=self.controller_configs,
                gripper_types=self.gripper_types,
                initialization_noise=self.initialization_noise,
                use_camera_obs=self.use_camera_obs,
                use_object_obs=self.use_object_obs,
                reward_scale=self.reward_scale,
                reward_shaping=self.reward_shaping,
                has_renderer=self.has_renderer,
                has_offscreen_renderer=self.has_offscreen_renderer,
                render_camera=self.render_camera,
                render_collision_mesh=self.render_collision_mesh,
                render_visual_mesh=self.render_visual_mesh,
                render_gpu_device_id=self.render_gpu_device_id,
                control_freq=self.control_freq,
                horizon=self.horizon,
                ignore_done=self.ignore_done,
                hard_reset=self.hard_reset,
                camera_names=self.camera_names,
                camera_heights=self.camera_heights,
                camera_widths=self.camera_widths,
                camera_depths=self.camera_depths,
                camera_segmentations=self.camera_segmentations,
                task_config=self.task_config,
                renderer=self.renderer,
                renderer_config=self.renderer_config,
            )
        )
