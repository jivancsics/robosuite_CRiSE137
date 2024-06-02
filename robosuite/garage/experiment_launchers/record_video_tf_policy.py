"""File for recording a video of a RL2-based trained policy on CRiSE 1/3 tasks as well as on CRiSE 7 tasks.
Also offers the option to take a screenshot of the initial task setup by setting argument 'screenshot_mode' in
function record_tf_policy to 'True'.
"""

from garage.experiment import Snapshotter
import tensorflow as tf
import robosuite as suite
import numpy as np
import imageio
import robosuite.utils.macros as macros
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.models.objects import BoxObject
from robosuite.utils.mjcf_utils import CustomMaterial
from matplotlib import pyplot as plt
import os

os.environ["IMAGEIO_FFMPEG_EXE"] = "ffmpeg"

# Setting image convention to OpenCV to make sure that the recorded video has a correct orientation
macros.IMAGE_CONVENTION = "opencv"


# Basic flatten_obs() function taken from gym_wrapper.py --> modified so that the necessary camera obs for video
# recording gets deleted from the overall observation after taking the env.step
def flatten_obs(obs_dict, env, single_task_ml, verbose=False):
    """
    Filters keys of interest out and concatenate the information.

    Args:
        obs_dict (OrderedDict): ordered dictionary of observations
        env (Robosuite Manipulation Environment): actual Robosuite environment
        single_task_ml (bool): Whether the used env is a CRiSE 1/3 task or not
        verbose (bool): Whether to print out to console as observation keys are processed

    Returns:
        np.array: observations flattened into a 1d array
    """
    env_name = str(env)
    keys = ["object-state", "robot0_proprio-state", "frontview_image"]
    ob_lst = []
    ob_lengths = []
    ob_lut = {}
    observation_dim = 33
    if 'IIWA14_extended_nolinear' in env.robot_names:
        robot_state_dim = 40
    else:
        robot_state_dim = 32

    if single_task_ml:
        ob_lst = []
        for key in keys:
            if key in obs_dict:
                if "frontview_image" in key:
                    break
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(np.array(obs_dict[key]).flatten())
        return np.concatenate(ob_lst)

    else:
        for i, key in enumerate(keys):
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lut[key] = i
                ob_lst.append(np.array(obs_dict[key]).flatten())
                ob_lengths.append(obs_dict[key].size)
                if i == 1:
                    if ob_lengths[i] == robot_state_dim:
                        ob_lst = ob_lst[i:] + ob_lst[:i]
                    break

        if 'robosuite.environments.manipulation.door.Door' in env_name:
            ob_lst[1] = np.insert(ob_lst[1], [3, 3, 3, 3], [0, 0, 0, 0])  # door quat not known
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
            raise Exception("No known CRiSE Meta-Reinforcement Learning task found!")

        return np.concatenate(ob_lst)


def record_tf_policy(screenshot_mode=False):
    """
    Function for recording the resulting interactions of a Tensorflow-based policy (RL2) within the task environment.

    Args:
        screenshot_mode (bool): If True, a screenshot of the initial task setup is taken and stored as a .png image

    """

    tf.compat.v1.disable_eager_execution()
    horizon = 500

    print("Welcome to the MRL Policy Recorder for Meta Learning across tasks (CRiSE 7) and single task MRL (CRiSE 1/3)!")
    print("Based on the RL2-learned policy, choose between the Rethink Robotics Sawyer and the Kuka "
          "IIWA14 with fixed position on the linear axes")
    print("[1] Rethink Robotics Sawyer")
    print("[2] Kuka IIWA14 with fixed position on the linear axes")
    choice_robot = input("Enter your number of choice: ")
    choice_robot = int(choice_robot)

    if choice_robot == 1:
        robot = "SAWYER"
    else:
        robot = "IIWA14"

    print("Choose between single task MRL [1] (=CRiSE 1/3) and MRL across tasks [2] (=CRiSE 7)")
    choice_metalearning = input("Enter your number of choice: ")
    choice_metalearning = int(choice_metalearning)

    if choice_metalearning == 1:

        print("Please enter the number of your learned CRiSE 1 task (Choose [2] for CRiSE 3):")
        print("[1] Open the Door")
        print("[2] Lift a block")
        print("[3] Round nut assembly")
        print("[4] Square nut assembly")
        print("[5] Mixed nut assembly")
        print("[6] Pick and place milk")
        print("[7] Pick and place cereal")
        print("[8] Pick and place can")
        print("[9] Pick and place bread")
        print("[10] Stack blocks")

        choice = input("Enter your number of choice: ")
        choice = int(choice)

        print("\033[92m {}\033[00m".format("CRiSE 1/3 POLICY RECORDING"))
        print("\033[92m {}\033[00m".format(robot))

        if choice_robot == 1:

            if choice == 1:

                placement_initializer = UniformRandomSampler(
                    name="ObjectSampler",
                    x_range=[0.08, 0.08],
                    y_range=[0, 0],
                    rotation=-np.pi / 2.0 - 0.1,
                    rotation_axis="z",
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=np.array((-0.2, -0.35, 0.8)),
                )

                env = suite.make(
                    env_name="Door",
                    robots="Sawyer",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    placement_initializer=placement_initializer,
                    use_object_obs=True,
                    use_latch=True,
                    hard_reset=False,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "OpenDoor"

            elif choice == 2:

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

                env = suite.make(
                    env_name="Lift",
                    robots="Sawyer",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    use_object_obs=True,
                    placement_initializer=placement_initializer,
                    hard_reset=False,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "LiftBlock"

            elif choice == 3:

                x_range_square = [0.12, 0.12]
                y_range_square = [0.15, 0.15]
                rotation_square = 0
                x_range_round = [-0.12, -0.12]
                y_range_round = [-0.15, -0.15]
                rotation_round = 0

                placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

                # Add square nut and round nut to the sequential object sampler
                placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name="SquareNutSampler",
                        x_range=x_range_square,
                        y_range=y_range_square,
                        rotation=rotation_square,
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=np.array((0, 0, 0.82)),
                        z_offset=0.02,
                    )
                )

                placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name="RoundNutSampler",
                        x_range=x_range_round,
                        y_range=y_range_round,
                        rotation=rotation_round,
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=np.array((0, 0, 0.82)),
                        z_offset=0.02,
                    )
                )

                env = suite.make(
                    env_name="NutAssemblyRound",
                    robots="Sawyer",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    placement_initializer=placement_initializer,
                    hard_reset=False,
                    use_object_obs=True,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "NutAssemblyRound"

            elif choice == 4:

                x_range_square = [-0.12, -0.12]
                y_range_square = [0.15, 0.15]
                rotation_square = 0
                x_range_round = [0.12, 0.12]
                y_range_round = [-0.15, -0.15]
                rotation_round = 0

                placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

                # Add square nut and round nut to the sequential object sampler
                placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name="SquareNutSampler",
                        x_range=x_range_square,
                        y_range=y_range_square,
                        rotation=rotation_square,
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=np.array((0, 0, 0.82)),
                        z_offset=0.02,
                    )
                )

                placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name="RoundNutSampler",
                        x_range=x_range_round,
                        y_range=y_range_round,
                        rotation=rotation_round,
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=np.array((0, 0, 0.82)),
                        z_offset=0.02,
                    )
                )

                env = suite.make(
                    env_name="NutAssemblySquare",
                    robots="Sawyer",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    placement_initializer=placement_initializer,
                    hard_reset=False,
                    use_object_obs=True,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "NutAssemblySquare"

            elif choice == 5:

                x_range_square = [-0.12, -0.12]
                y_range_square = [0.15, 0.15]
                rotation_square = 0
                x_range_round = [-0.12, -0.12]
                y_range_round = [-0.15, -0.15]
                rotation_round = 0

                placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

                # Add square nut and round nut to the sequential object sampler
                placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name="SquareNutSampler",
                        x_range=x_range_square,
                        y_range=y_range_square,
                        rotation=rotation_square,
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=np.array((0, 0, 0.82)),
                        z_offset=0.02,
                    )
                )

                placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name="RoundNutSampler",
                        x_range=x_range_round,
                        y_range=y_range_round,
                        rotation=rotation_round,
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=np.array((0, 0, 0.82)),
                        z_offset=0.02,
                    )
                )

                env = suite.make(
                    env_name="NutAssembly",
                    robots="Sawyer",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    placement_initializer=placement_initializer,
                    single_object_mode=0,
                    nut_type=None,
                    hard_reset=False,
                    use_object_obs=True,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "NutAssemblyMixed"

            elif choice == 6:

                env = suite.make(
                    env_name="PickPlaceMilk",
                    robots="Sawyer",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    bin1_pos=[-0.1, -0.27, 0.8],
                    bin2_pos=[0.1, 0.3, 0.8],
                    hard_reset=False,
                    use_object_obs=True,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "PickPlaceMilk"

            elif choice == 7:

                env = suite.make(
                    env_name="PickPlaceCereal",
                    robots="Sawyer",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    bin1_pos=[-0.1, -0.27, 0.8],
                    bin2_pos=[0.1, 0.3, 0.8],
                    hard_reset=False,
                    use_object_obs=True,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "PickPlaceCereal"

            elif choice == 8:

                env = suite.make(
                    env_name="PickPlaceCan",
                    robots="Sawyer",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    bin1_pos=[-0.1, -0.27, 0.8],
                    bin2_pos=[0.1, 0.3, 0.8],
                    hard_reset=False,
                    use_object_obs=True,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "PickPlaceCan"

            elif choice == 9:

                env = suite.make(
                    env_name="PickPlaceBread",
                    robots="Sawyer",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    bin1_pos=[-0.1, -0.27, 0.8],
                    bin2_pos=[0.1, 0.3, 0.8],
                    hard_reset=False,
                    use_object_obs=True,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "PickPlaceBread"

            elif choice == 10:
                # initialize the two boxes
                tex_attrib = {
                    "type": "cube",
                }
                mat_attrib = {
                    "texrepeat": "1 1",
                    "specular": "0.4",
                    "shininess": "0.1",
                }
                redwood = CustomMaterial(
                    texture="WoodRed",
                    tex_name="redwood",
                    mat_name="redwood_mat",
                    tex_attrib=tex_attrib,
                    mat_attrib=mat_attrib,
                )
                greenwood = CustomMaterial(
                    texture="WoodGreen",
                    tex_name="greenwood",
                    mat_name="greenwood_mat",
                    tex_attrib=tex_attrib,
                    mat_attrib=mat_attrib,
                )

                cubeA = BoxObject(
                    name="cubeA",
                    size_min=[0.02, 0.02, 0.02],
                    size_max=[0.02, 0.02, 0.02],
                    rgba=[1, 0, 0, 1],
                    material=redwood,
                )
                cubeB = BoxObject(
                    name="cubeB",
                    size_min=[0.025, 0.025, 0.025],
                    size_max=[0.025, 0.025, 0.025],
                    rgba=[0, 1, 0, 1],
                    material=greenwood,
                )

                placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

                placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name="CubeASampler",
                        x_range=[0.3, 0.3],
                        y_range=[0.3, 0.3],
                        rotation=0,
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=np.array((0, 0, 0.8)),
                        z_offset=0.01,
                    )
                )

                placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name="CubeBSampler",
                        x_range=[0.0, 0.0],
                        y_range=[0.0, 0.0],
                        rotation=0,
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=np.array((0, 0, 0.8)),
                        z_offset=0.01,
                    )
                )

                placement_initializer.add_objects_to_sampler(sampler_name="CubeASampler", mujoco_objects=cubeA)
                placement_initializer.add_objects_to_sampler(sampler_name="CubeBSampler", mujoco_objects=cubeB)

                env = suite.make(
                    env_name="Stack",
                    robots="Sawyer",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    placement_initializer=placement_initializer,
                    hard_reset=False,
                    use_object_obs=True,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "StackBlocks"

            else:
                raise Exception("Error! Please enter an integer number in the range 1 to 10!")

        elif choice_robot == 2:

            if choice == 1:

                placement_initializer = UniformRandomSampler(
                    name="ObjectSampler",
                    x_range=[0.08, 0.08],
                    y_range=[0, 0],
                    rotation=-np.pi / 2.0 - 0.1,
                    rotation_axis="z",
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=np.array((-0.2, -0.35, 0.8)),
                )

                env = suite.make(
                    env_name="Door",
                    robots="IIWA14_extended_nolinear",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    placement_initializer=placement_initializer,
                    use_object_obs=True,
                    use_latch=True,
                    hard_reset=False,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "OpenDoor"

            elif choice == 2:

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

                env = suite.make(
                    env_name="Lift",
                    robots="IIWA14_extended_nolinear",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    use_object_obs=True,
                    placement_initializer=placement_initializer,
                    hard_reset=False,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "LiftBlock"

            elif choice == 3:

                x_range_square = [0.12, 0.12]
                y_range_square = [0.15, 0.15]
                rotation_square = 0
                x_range_round = [-0.12, -0.12]
                y_range_round = [-0.15, -0.15]
                rotation_round = 0

                placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

                # Add square nut and round nut to the sequential object sampler
                placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name="SquareNutSampler",
                        x_range=x_range_square,
                        y_range=y_range_square,
                        rotation=rotation_square,
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=np.array((0, 0, 0.82)),
                        z_offset=0.02,
                    )
                )

                placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name="RoundNutSampler",
                        x_range=x_range_round,
                        y_range=y_range_round,
                        rotation=rotation_round,
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=np.array((0, 0, 0.82)),
                        z_offset=0.02,
                    )
                )

                env = suite.make(
                    env_name="NutAssemblyRound",
                    robots="IIWA14_extended_nolinear",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    placement_initializer=placement_initializer,
                    hard_reset=False,
                    use_object_obs=True,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "NutAssemblyRound"

            elif choice == 4:

                x_range_square = [-0.12, -0.12]
                y_range_square = [0.15, 0.15]
                rotation_square = 0
                x_range_round = [0.12, 0.12]
                y_range_round = [-0.15, -0.15]
                rotation_round = 0

                placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

                # Add square nut and round nut to the sequential object sampler
                placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name="SquareNutSampler",
                        x_range=x_range_square,
                        y_range=y_range_square,
                        rotation=rotation_square,
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=np.array((0, 0, 0.82)),
                        z_offset=0.02,
                    )
                )

                placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name="RoundNutSampler",
                        x_range=x_range_round,
                        y_range=y_range_round,
                        rotation=rotation_round,
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=np.array((0, 0, 0.82)),
                        z_offset=0.02,
                    )
                )

                env = suite.make(
                    env_name="NutAssemblySquare",
                    robots="IIWA14_extended_nolinear",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    placement_initializer=placement_initializer,
                    hard_reset=False,
                    use_object_obs=True,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "NutAssemblySquare"

            elif choice == 5:

                x_range_square = [-0.12, -0.12]
                y_range_square = [0.15, 0.15]
                rotation_square = 0
                x_range_round = [-0.12, -0.12]
                y_range_round = [-0.15, -0.15]
                rotation_round = 0

                placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

                # Add square nut and round nut to the sequential object sampler
                placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name="SquareNutSampler",
                        x_range=x_range_square,
                        y_range=y_range_square,
                        rotation=rotation_square,
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=np.array((0, 0, 0.82)),
                        z_offset=0.02,
                    )
                )

                placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name="RoundNutSampler",
                        x_range=x_range_round,
                        y_range=y_range_round,
                        rotation=rotation_round,
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=np.array((0, 0, 0.82)),
                        z_offset=0.02,
                    )
                )

                env = suite.make(
                    env_name="NutAssembly",
                    robots="IIWA14_extended_nolinear",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    placement_initializer=placement_initializer,
                    single_object_mode=0,
                    nut_type=None,
                    hard_reset=False,
                    use_object_obs=True,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "NutAssemblyMixed"

            elif choice == 6:

                env = suite.make(
                    env_name="PickPlaceMilk",
                    robots="IIWA14_extended_nolinear",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    bin1_pos=[-0.1, -0.27, 0.8],
                    bin2_pos=[0.1, 0.3, 0.8],
                    hard_reset=False,
                    use_object_obs=True,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "PickPlaceMilk"

            elif choice == 7:

                env = suite.make(
                    env_name="PickPlaceCereal",
                    robots="IIWA14_extended_nolinear",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    bin1_pos=[-0.1, -0.27, 0.8],
                    bin2_pos=[0.1, 0.3, 0.8],
                    hard_reset=False,
                    use_object_obs=True,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "PickPlaceCereal"

            elif choice == 8:

                env = suite.make(
                    env_name="PickPlaceCan",
                    robots="IIWA14_extended_nolinear",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    bin1_pos=[-0.1, -0.27, 0.8],
                    bin2_pos=[0.1, 0.3, 0.8],
                    hard_reset=False,
                    use_object_obs=True,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "PickPlaceCan"

            elif choice == 9:

                env = suite.make(
                    env_name="PickPlaceBread",
                    robots="IIWA14_extended_nolinear",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    bin1_pos=[-0.1, -0.27, 0.8],
                    bin2_pos=[0.1, 0.3, 0.8],
                    hard_reset=False,
                    use_object_obs=True,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "PickPlaceBread"

            elif choice == 10:
                # initialize the two boxes
                tex_attrib = {
                    "type": "cube",
                }
                mat_attrib = {
                    "texrepeat": "1 1",
                    "specular": "0.4",
                    "shininess": "0.1",
                }
                redwood = CustomMaterial(
                    texture="WoodRed",
                    tex_name="redwood",
                    mat_name="redwood_mat",
                    tex_attrib=tex_attrib,
                    mat_attrib=mat_attrib,
                )
                greenwood = CustomMaterial(
                    texture="WoodGreen",
                    tex_name="greenwood",
                    mat_name="greenwood_mat",
                    tex_attrib=tex_attrib,
                    mat_attrib=mat_attrib,
                )

                cubeA = BoxObject(
                    name="cubeA",
                    size_min=[0.02, 0.02, 0.02],
                    size_max=[0.02, 0.02, 0.02],
                    rgba=[1, 0, 0, 1],
                    material=redwood,
                )
                cubeB = BoxObject(
                    name="cubeB",
                    size_min=[0.025, 0.025, 0.025],
                    size_max=[0.025, 0.025, 0.025],
                    rgba=[0, 1, 0, 1],
                    material=greenwood,
                )

                placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

                placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name="CubeASampler",
                        x_range=[0.3, 0.3],
                        y_range=[0.3, 0.3],
                        rotation=0,
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=np.array((0, 0, 0.8)),
                        z_offset=0.01,
                    )
                )

                placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name="CubeBSampler",
                        x_range=[0.0, 0.0],
                        y_range=[0.0, 0.0],
                        rotation=0,
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=np.array((0, 0, 0.8)),
                        z_offset=0.01,
                    )
                )

                placement_initializer.add_objects_to_sampler(sampler_name="CubeASampler", mujoco_objects=cubeA)
                placement_initializer.add_objects_to_sampler(sampler_name="CubeBSampler", mujoco_objects=cubeB)

                env = suite.make(
                    env_name="Stack",
                    robots="IIWA14_extended_nolinear",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    placement_initializer=placement_initializer,
                    hard_reset=False,
                    use_object_obs=True,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "StackBlocks"

            else:
                raise Exception("Error! Please enter an integer number in the range 1 to 10!")

        else:
            raise Exception("Robot Error! Please enter [1] for the Sawyer or [2] for the Kuka IIWA14 robot!")

        snapshotter = Snapshotter()
        video_writer = imageio.get_writer(name + '_' + robot + '_RL2_CRiSE1.mp4', mode='I', fps=20)
        with tf.compat.v1.Session():  # only for Tensorflow
            if choice_robot == 2:
                data = snapshotter.load('IIWA14_extended_nolinear/data/local/experiment/crise1_rl2_ppo')
            else:
                data = snapshotter.load('Sawyer/data/local/experiment/crise1_rl2_ppo')
            policy = data['algo'].policy
            obs_dict = env.reset()  # Initial observation

            if screenshot_mode:
                screenshot = obs_dict["frontview_image"]
                plt.imsave("{}.png".format(name), screenshot)

            obs = flatten_obs(obs_dict, env, True)
            obs = np.concatenate([obs, np.zeros(env.action_dim), [0], [0]])
            policy.reset()
            success_counter = 0

            for steps in range(horizon):
                action, _ = policy.get_action(obs)
                obs_dict, reward, done, info = env.step(action)

                # Record the frame with the video_writer
                frame = obs_dict["frontview_image"]
                video_writer.append_data(frame)
                print("Saving frame #{}".format(steps))

                # Use modified flatten_obs() to delete the camera observations and reorder/flatten
                # the obs array correctly
                obs = flatten_obs(obs_dict, env, True)

                # Concatenate RL2 specific information last action, last reward and actual step type
                # see garage/tf/algos/rl2.py wrapper for more information
                if steps != (horizon - 1):
                    obs = np.concatenate([obs, action, [reward], [0]])
                else:
                    obs = np.concatenate([obs, action, [reward], [1]])

                if info['success']:
                    success_counter += 1
                    if success_counter > 40:
                        return

            video_writer.close()
            env.close()

    elif choice_metalearning == 2:
        print("\033[92m {}\033[00m".format("CRiSE 7 POLICY RECORDING"))
        print("\033[92m {}\033[00m".format(robot))

        print("Please enter a number to record one of the CRiSE 7 tasks:")
        print("Train tasks:")
        print("------------")
        print("[1] Open the Door")
        print("[2] Lift a block")
        print("[3] Round nut assembly")
        print("[4] Mixed nut assembly")
        print("[5] Pick and place milk")
        print("[6] Pick and place cereal")
        print("[7] Pick and place bread")

        print("Test tasks:")
        print("-----------")
        print("[8] Pick and place can")
        print("[9] Square nut assembly")
        print("[10] Stack blocks")

        choice = input("Enter your number of choice: ")
        choice = int(choice)

        if choice_robot == 1:

            if choice == 1:

                placement_initializer = UniformRandomSampler(
                    name="ObjectSampler",
                    x_range=[0.08, 0.08],
                    y_range=[0, 0],
                    rotation=-np.pi / 2.0 - 0.1,
                    rotation_axis="z",
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=np.array((-0.2, -0.35, 0.8)),
                )

                env = suite.make(
                    env_name="Door",
                    robots="Sawyer",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    placement_initializer=placement_initializer,
                    use_object_obs=True,
                    use_latch=True,
                    hard_reset=False,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "OpenDoor"

            elif choice == 2:

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

                env = suite.make(
                    env_name="Lift",
                    robots="Sawyer",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    use_object_obs=True,
                    placement_initializer=placement_initializer,
                    hard_reset=False,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "LiftBlock"

            elif choice == 3:

                x_range_square = [0.12, 0.12]
                y_range_square = [0.15, 0.15]
                rotation_square = 0
                x_range_round = [-0.12, -0.12]
                y_range_round = [-0.15, -0.15]
                rotation_round = 0

                placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

                # Add square nut and round nut to the sequential object sampler
                placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name="SquareNutSampler",
                        x_range=x_range_square,
                        y_range=y_range_square,
                        rotation=rotation_square,
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=np.array((0, 0, 0.82)),
                        z_offset=0.02,
                    )
                )

                placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name="RoundNutSampler",
                        x_range=x_range_round,
                        y_range=y_range_round,
                        rotation=rotation_round,
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=np.array((0, 0, 0.82)),
                        z_offset=0.02,
                    )
                )

                env = suite.make(
                    env_name="NutAssemblyRound",
                    robots="Sawyer",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    placement_initializer=placement_initializer,
                    hard_reset=False,
                    use_object_obs=True,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "NutAssemblyRound"

            elif choice == 9:

                x_range_square = [-0.12, -0.12]
                y_range_square = [0.15, 0.15]
                rotation_square = 0
                x_range_round = [0.12, 0.12]
                y_range_round = [-0.15, -0.15]
                rotation_round = 0

                placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

                # Add square nut and round nut to the sequential object sampler
                placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name="SquareNutSampler",
                        x_range=x_range_square,
                        y_range=y_range_square,
                        rotation=rotation_square,
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=np.array((0, 0, 0.82)),
                        z_offset=0.02,
                    )
                )

                placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name="RoundNutSampler",
                        x_range=x_range_round,
                        y_range=y_range_round,
                        rotation=rotation_round,
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=np.array((0, 0, 0.82)),
                        z_offset=0.02,
                    )
                )

                env = suite.make(
                    env_name="NutAssemblySquare",
                    robots="Sawyer",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    placement_initializer=placement_initializer,
                    hard_reset=False,
                    use_object_obs=True,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "NutAssemblySquare"

            elif choice == 4:

                x_range_square = [-0.12, -0.12]
                y_range_square = [0.15, 0.15]
                rotation_square = 0
                x_range_round = [-0.12, -0.12]
                y_range_round = [-0.15, -0.15]
                rotation_round = 0

                placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

                # Add square nut and round nut to the sequential object sampler
                placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name="SquareNutSampler",
                        x_range=x_range_square,
                        y_range=y_range_square,
                        rotation=rotation_square,
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=np.array((0, 0, 0.82)),
                        z_offset=0.02,
                    )
                )

                placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name="RoundNutSampler",
                        x_range=x_range_round,
                        y_range=y_range_round,
                        rotation=rotation_round,
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=np.array((0, 0, 0.82)),
                        z_offset=0.02,
                    )
                )

                env = suite.make(
                    env_name="NutAssembly",
                    robots="Sawyer",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    placement_initializer=placement_initializer,
                    single_object_mode=0,
                    nut_type=None,
                    hard_reset=False,
                    use_object_obs=True,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "NutAssemblyMixed"

            elif choice == 5:

                env = suite.make(
                    env_name="PickPlaceMilk",
                    robots="Sawyer",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    bin1_pos=[-0.1, -0.27, 0.8],
                    bin2_pos=[0.1, 0.3, 0.8],
                    hard_reset=False,
                    use_object_obs=True,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "PickPlaceMilk"

            elif choice == 6:

                env = suite.make(
                    env_name="PickPlaceCereal",
                    robots="Sawyer",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    bin1_pos=[-0.1, -0.27, 0.8],
                    bin2_pos=[0.1, 0.3, 0.8],
                    hard_reset=False,
                    use_object_obs=True,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "PickPlaceCereal"

            elif choice == 8:

                env = suite.make(
                    env_name="PickPlaceCan",
                    robots="Sawyer",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    bin1_pos=[-0.1, -0.27, 0.8],
                    bin2_pos=[0.1, 0.3, 0.8],
                    hard_reset=False,
                    use_object_obs=True,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "PickPlaceCan"

            elif choice == 7:

                env = suite.make(
                    env_name="PickPlaceBread",
                    robots="Sawyer",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    bin1_pos=[-0.1, -0.27, 0.8],
                    bin2_pos=[0.1, 0.3, 0.8],
                    hard_reset=False,
                    use_object_obs=True,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "PickPlaceBread"

            elif choice == 10:
                # initialize the two boxes
                tex_attrib = {
                    "type": "cube",
                }
                mat_attrib = {
                    "texrepeat": "1 1",
                    "specular": "0.4",
                    "shininess": "0.1",
                }
                redwood = CustomMaterial(
                    texture="WoodRed",
                    tex_name="redwood",
                    mat_name="redwood_mat",
                    tex_attrib=tex_attrib,
                    mat_attrib=mat_attrib,
                )
                greenwood = CustomMaterial(
                    texture="WoodGreen",
                    tex_name="greenwood",
                    mat_name="greenwood_mat",
                    tex_attrib=tex_attrib,
                    mat_attrib=mat_attrib,
                )

                cubeA = BoxObject(
                    name="cubeA",
                    size_min=[0.02, 0.02, 0.02],
                    size_max=[0.02, 0.02, 0.02],
                    rgba=[1, 0, 0, 1],
                    material=redwood,
                )
                cubeB = BoxObject(
                    name="cubeB",
                    size_min=[0.025, 0.025, 0.025],
                    size_max=[0.025, 0.025, 0.025],
                    rgba=[0, 1, 0, 1],
                    material=greenwood,
                )

                placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

                placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name="CubeASampler",
                        x_range=[0.3, 0.3],
                        y_range=[0.3, 0.3],
                        rotation=0,
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=np.array((0, 0, 0.8)),
                        z_offset=0.01,
                    )
                )

                placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name="CubeBSampler",
                        x_range=[0.0, 0.0],
                        y_range=[0.0, 0.0],
                        rotation=0,
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=np.array((0, 0, 0.8)),
                        z_offset=0.01,
                    )
                )

                placement_initializer.add_objects_to_sampler(sampler_name="CubeASampler", mujoco_objects=cubeA)
                placement_initializer.add_objects_to_sampler(sampler_name="CubeBSampler", mujoco_objects=cubeB)

                env = suite.make(
                    env_name="Stack",
                    robots="Sawyer",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    placement_initializer=placement_initializer,
                    hard_reset=False,
                    use_object_obs=True,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "StackBlocks"

            else:
                raise Exception("Error! Please enter an integer number in the range 1 to 10!")

        elif choice_robot == 2:

            if choice == 1:

                placement_initializer = UniformRandomSampler(
                    name="ObjectSampler",
                    x_range=[0.08, 0.08],
                    y_range=[0, 0],
                    rotation=-np.pi / 2.0 - 0.1,
                    rotation_axis="z",
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=np.array((-0.2, -0.35, 0.8)),
                )

                env = suite.make(
                    env_name="Door",
                    robots="IIWA14_extended_nolinear",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    placement_initializer=placement_initializer,
                    use_object_obs=True,
                    use_latch=True,
                    hard_reset=False,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "OpenDoor"

            elif choice == 2:

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

                env = suite.make(
                    env_name="Lift",
                    robots="IIWA14_extended_nolinear",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    use_object_obs=True,
                    placement_initializer=placement_initializer,
                    hard_reset=False,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "LiftBlock"

            elif choice == 3:

                x_range_square = [0.12, 0.12]
                y_range_square = [0.15, 0.15]
                rotation_square = 0
                x_range_round = [-0.12, -0.12]
                y_range_round = [-0.15, -0.15]
                rotation_round = 0

                placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

                # Add square nut and round nut to the sequential object sampler
                placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name="SquareNutSampler",
                        x_range=x_range_square,
                        y_range=y_range_square,
                        rotation=rotation_square,
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=np.array((0, 0, 0.82)),
                        z_offset=0.02,
                    )
                )

                placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name="RoundNutSampler",
                        x_range=x_range_round,
                        y_range=y_range_round,
                        rotation=rotation_round,
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=np.array((0, 0, 0.82)),
                        z_offset=0.02,
                    )
                )

                env = suite.make(
                    env_name="NutAssemblyRound",
                    robots="IIWA14_extended_nolinear",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    placement_initializer=placement_initializer,
                    hard_reset=False,
                    use_object_obs=True,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "NutAssemblyRound"

            elif choice == 9:

                x_range_square = [-0.12, -0.12]
                y_range_square = [0.15, 0.15]
                rotation_square = 0
                x_range_round = [0.12, 0.12]
                y_range_round = [-0.15, -0.15]
                rotation_round = 0

                placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

                # Add square nut and round nut to the sequential object sampler
                placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name="SquareNutSampler",
                        x_range=x_range_square,
                        y_range=y_range_square,
                        rotation=rotation_square,
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=np.array((0, 0, 0.82)),
                        z_offset=0.02,
                    )
                )

                placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name="RoundNutSampler",
                        x_range=x_range_round,
                        y_range=y_range_round,
                        rotation=rotation_round,
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=np.array((0, 0, 0.82)),
                        z_offset=0.02,
                    )
                )

                env = suite.make(
                    env_name="NutAssemblySquare",
                    robots="IIWA14_extended_nolinear",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    placement_initializer=placement_initializer,
                    hard_reset=False,
                    use_object_obs=True,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "NutAssemblySquare"

            elif choice == 4:

                x_range_square = [-0.12, -0.12]
                y_range_square = [0.15, 0.15]
                rotation_square = 0
                x_range_round = [-0.12, -0.12]
                y_range_round = [-0.15, -0.15]
                rotation_round = 0

                placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

                # Add square nut and round nut to the sequential object sampler
                placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name="SquareNutSampler",
                        x_range=x_range_square,
                        y_range=y_range_square,
                        rotation=rotation_square,
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=np.array((0, 0, 0.82)),
                        z_offset=0.02,
                    )
                )

                placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name="RoundNutSampler",
                        x_range=x_range_round,
                        y_range=y_range_round,
                        rotation=rotation_round,
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=np.array((0, 0, 0.82)),
                        z_offset=0.02,
                    )
                )

                env = suite.make(
                    env_name="NutAssembly",
                    robots="IIWA14_extended_nolinear",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    placement_initializer=placement_initializer,
                    single_object_mode=0,
                    nut_type=None,
                    hard_reset=False,
                    use_object_obs=True,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "NutAssemblyMixed"

            elif choice == 5:

                env = suite.make(
                    env_name="PickPlaceMilk",
                    robots="IIWA14_extended_nolinear",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    bin1_pos=[-0.1, -0.27, 0.8],
                    bin2_pos=[0.1, 0.3, 0.8],
                    hard_reset=False,
                    use_object_obs=True,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "PickPlaceMilk"

            elif choice == 6:

                env = suite.make(
                    env_name="PickPlaceCereal",
                    robots="IIWA14_extended_nolinear",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    bin1_pos=[-0.1, -0.27, 0.8],
                    bin2_pos=[0.1, 0.3, 0.8],
                    hard_reset=False,
                    use_object_obs=True,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "PickPlaceCereal"

            elif choice == 8:

                env = suite.make(
                    env_name="PickPlaceCan",
                    robots="IIWA14_extended_nolinear",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    bin1_pos=[-0.1, -0.27, 0.8],
                    bin2_pos=[0.1, 0.3, 0.8],
                    hard_reset=False,
                    use_object_obs=True,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "PickPlaceCan"

            elif choice == 7:

                env = suite.make(
                    env_name="PickPlaceBread",
                    robots="IIWA14_extended_nolinear",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    bin1_pos=[-0.1, -0.27, 0.8],
                    bin2_pos=[0.1, 0.3, 0.8],
                    hard_reset=False,
                    use_object_obs=True,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "PickPlaceBread"

            elif choice == 10:
                # initialize the two boxes
                tex_attrib = {
                    "type": "cube",
                }
                mat_attrib = {
                    "texrepeat": "1 1",
                    "specular": "0.4",
                    "shininess": "0.1",
                }
                redwood = CustomMaterial(
                    texture="WoodRed",
                    tex_name="redwood",
                    mat_name="redwood_mat",
                    tex_attrib=tex_attrib,
                    mat_attrib=mat_attrib,
                )
                greenwood = CustomMaterial(
                    texture="WoodGreen",
                    tex_name="greenwood",
                    mat_name="greenwood_mat",
                    tex_attrib=tex_attrib,
                    mat_attrib=mat_attrib,
                )

                cubeA = BoxObject(
                    name="cubeA",
                    size_min=[0.02, 0.02, 0.02],
                    size_max=[0.02, 0.02, 0.02],
                    rgba=[1, 0, 0, 1],
                    material=redwood,
                )
                cubeB = BoxObject(
                    name="cubeB",
                    size_min=[0.025, 0.025, 0.025],
                    size_max=[0.025, 0.025, 0.025],
                    rgba=[0, 1, 0, 1],
                    material=greenwood,
                )

                placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

                placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name="CubeASampler",
                        x_range=[0.3, 0.3],
                        y_range=[0.3, 0.3],
                        rotation=0,
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=np.array((0, 0, 0.8)),
                        z_offset=0.01,
                    )
                )

                placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name="CubeBSampler",
                        x_range=[0.0, 0.0],
                        y_range=[0.0, 0.0],
                        rotation=0,
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=np.array((0, 0, 0.8)),
                        z_offset=0.01,
                    )
                )

                placement_initializer.add_objects_to_sampler(sampler_name="CubeASampler", mujoco_objects=cubeA)
                placement_initializer.add_objects_to_sampler(sampler_name="CubeBSampler", mujoco_objects=cubeB)

                env = suite.make(
                    env_name="Stack",
                    robots="IIWA14_extended_nolinear",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    placement_initializer=placement_initializer,
                    hard_reset=False,
                    use_object_obs=True,
                    camera_names="frontview",
                    camera_widths=1024,
                    camera_heights=1024,
                )
                name = "StackBlocks"

            else:
                raise Exception("Error! Please enter an integer number in the range 1 to 10!")

        else:
            raise Exception("Robot Error! Please enter [1] for the Sawyer or [2] for the Kuka IIWA14 robot!")

        snapshotter = Snapshotter()
        video_writer = imageio.get_writer(name + '_' + robot + '_RL2_CRiSE7.mp4', mode='I', fps=20)
        with tf.compat.v1.Session():  # only for Tensorflow
            if choice_robot == 2:
                data = snapshotter.load('IIWA14_extended_nolinear/data/local/experiment/crise7_rl2_ppo')
            else:
                data = snapshotter.load('Sawyer/data/local/experiment/crise7_rl2_ppo')
            policy = data['algo'].policy
            obs_dict = env.reset()  # Initial observation

            if screenshot_mode:
                screenshot = obs_dict["frontview_image"]
                plt.imsave("{}.png".format(name), screenshot)

            obs = flatten_obs(obs_dict, env, False)
            obs = np.concatenate([obs, np.zeros(env.action_dim), [0], [0]])
            policy.reset()
            success_counter = 0

            for steps in range(horizon):
                action, _ = policy.get_action(obs)
                obs_dict, reward, done, info = env.step(action)

                # Record the frame with the video_writer
                frame = obs_dict["frontview_image"]
                video_writer.append_data(frame)
                print("Saving frame #{}".format(steps))

                # Use modified flatten_obs() to delete the camera observations and reorder/flatten
                # the obs array correctly
                obs = flatten_obs(obs_dict, env, False)

                # Concatenate RL2 specific information last action, last reward and actual step type
                # see garage/tf/algos/rl2.py wrapper for more information
                if steps != (horizon - 1):
                    obs = np.concatenate([obs, action, [reward], [0]])
                else:
                    obs = np.concatenate([obs, action, [reward], [1]])

                if info['success']:
                    success_counter += 1
                    if success_counter > 40:
                        return

            video_writer.close()
            env.close()

    else:
        raise Exception("Error! Please enter [1] for CRiSE 1/3 or [2] for CRiSE 7 MRL policy recording!")


if __name__ == "__main__":
    record_tf_policy(screenshot_mode=True)
