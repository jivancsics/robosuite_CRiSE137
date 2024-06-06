import robosuite as suite
import numpy as np
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.models.objects import BoxObject
from robosuite.utils.mjcf_utils import CustomMaterial

if __name__ == "__main__":

    horizon = 500
    while True:
        print("Welcome to the Robosuite CRiSE Task Viewer!")
        print("First choose between the Rethink Robotics Sawyer and the Kuka IIWA14 with fixed position on the linear axes")
        print("[1] Rethink Robotics Sawyer")
        print("[2] Kuka IIWA14 with fixed position on the linear axes")
        choice_robot = input("Enter your number of choice: ")
        choice_robot = int(choice_robot)

        print("Please enter a number to see one of the following tasks:")
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
                    has_renderer=True,
                    has_offscreen_renderer=False,
                    use_camera_obs=False,
                    placement_initializer=placement_initializer,
                    use_object_obs=True,
                    use_latch=True,
                )

            elif choice == 2:

                placement_initializer = UniformRandomSampler(
                    name="ObjectSampler",
                    x_range=[-0.1, -0.1],    # dimension of the table: (0.8, 0.8, 0.05) --> sampling range 1/2 of its surface
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
                    has_renderer=True,
                    has_offscreen_renderer=False,
                    use_camera_obs=False,
                    placement_initializer=placement_initializer,
                )

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
                    has_renderer=True,
                    has_offscreen_renderer=False,
                    use_camera_obs=False,
                    placement_initializer=placement_initializer,
                )

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
                    has_renderer=True,
                    has_offscreen_renderer=False,
                    use_camera_obs=False,
                    placement_initializer=placement_initializer,
                )

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
                    has_renderer=True,
                    has_offscreen_renderer=False,
                    use_camera_obs=False,
                    placement_initializer=placement_initializer,
                    single_object_mode=0,
                    nut_type=None,
                )

            elif choice == 6:

                env = suite.make(
                    env_name="PickPlaceMilk",
                    robots="Sawyer",
                    has_renderer=True,
                    has_offscreen_renderer=False,
                    use_camera_obs=False,
                    bin1_pos=[-0.1, -0.27, 0.8],
                    bin2_pos=[0.1, 0.3, 0.8],
                )

            elif choice == 7:

                env = suite.make(
                    env_name="PickPlaceCereal",
                    robots="Sawyer",
                    has_renderer=True,
                    has_offscreen_renderer=False,
                    use_camera_obs=False,
                    bin1_pos=[-0.1, -0.27, 0.8],
                    bin2_pos=[0.1, 0.3, 0.8],
                )

            elif choice == 8:

                env = suite.make(
                    env_name="PickPlaceCan",
                    robots="Sawyer",
                    has_renderer=True,
                    has_offscreen_renderer=False,
                    use_camera_obs=False,
                    bin1_pos=[-0.1, -0.27, 0.8],
                    bin2_pos=[0.1, 0.3, 0.8],
                )

            elif choice == 9:

                env = suite.make(
                    env_name="PickPlaceBread",
                    robots="Sawyer",
                    has_renderer=True,
                    has_offscreen_renderer=False,
                    use_camera_obs=False,
                    bin1_pos=[-0.1, -0.27, 0.8],
                    bin2_pos=[0.1, 0.3, 0.8],
                )

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
                    has_renderer=True,
                    has_offscreen_renderer=False,
                    use_camera_obs=False,
                    placement_initializer=placement_initializer,
                )

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
                    has_renderer=True,
                    has_offscreen_renderer=False,
                    use_camera_obs=False,
                    placement_initializer=placement_initializer,
                    use_object_obs=True,
                    use_latch=True,
                )

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
                    has_renderer=True,
                    has_offscreen_renderer=False,
                    use_camera_obs=False,
                    placement_initializer=placement_initializer,
                )

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
                    has_renderer=True,
                    has_offscreen_renderer=False,
                    use_camera_obs=False,
                    placement_initializer=placement_initializer,
                )

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
                    has_renderer=True,
                    has_offscreen_renderer=False,
                    use_camera_obs=False,
                    placement_initializer=placement_initializer,
                )

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
                    has_renderer=True,
                    has_offscreen_renderer=False,
                    use_camera_obs=False,
                    placement_initializer=placement_initializer,
                    single_object_mode=0,
                    nut_type=None,
                )

            elif choice == 6:

                env = suite.make(
                    env_name="PickPlaceMilk",
                    robots="IIWA14_extended_nolinear",
                    has_renderer=True,
                    has_offscreen_renderer=False,
                    use_camera_obs=False,
                    bin1_pos=[-0.1, -0.27, 0.8],
                    bin2_pos=[0.1, 0.3, 0.8],
                )

            elif choice == 7:

                env = suite.make(
                    env_name="PickPlaceCereal",
                    robots="IIWA14_extended_nolinear",
                    has_renderer=True,
                    has_offscreen_renderer=False,
                    use_camera_obs=False,
                    bin1_pos=[-0.1, -0.27, 0.8],
                    bin2_pos=[0.1, 0.3, 0.8],
                )

            elif choice == 8:

                env = suite.make(
                    env_name="PickPlaceCan",
                    robots="IIWA14_extended_nolinear",
                    has_renderer=True,
                    has_offscreen_renderer=False,
                    use_camera_obs=False,
                    bin1_pos=[-0.1, -0.27, 0.8],
                    bin2_pos=[0.1, 0.3, 0.8],
                )

            elif choice == 9:

                env = suite.make(
                    env_name="PickPlaceBread",
                    robots="IIWA14_extended_nolinear",
                    has_renderer=True,
                    has_offscreen_renderer=False,
                    use_camera_obs=False,
                    bin1_pos=[-0.1, -0.27, 0.8],
                    bin2_pos=[0.1, 0.3, 0.8],
                )

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
                    has_renderer=True,
                    has_offscreen_renderer=False,
                    use_camera_obs=False,
                    placement_initializer=placement_initializer,
                )

            else:
                raise Exception("Error! Please enter an integer number in the range 1 to 10!")

        else:
            raise Exception("Robot Error! Please enter [1] for the Sawyer or [2] for the Kuka IIWA14 robot!")

        # Reset the environment
        env.reset()
        list_info = []

        for i in range(500):
            action = np.random.randn(env.robots[0].dof)  # sample random action
            obs, reward, done, info = env.step(action)  # take action in the environment
            list_info.append(info)

            env.render()  # render on display
            if i == 250:
                env.reset()

        env.close()