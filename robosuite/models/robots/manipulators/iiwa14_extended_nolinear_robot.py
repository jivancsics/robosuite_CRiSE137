import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class IIWA14_extended_nolinear(ManipulatorModel):
    """
    IIWA14_extended is a bright and spunky robot created by KUKA mounted upside down on a frame. The linear axis are disabled in this model

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/iiwa14_extended_nolinear/robot.xml"), idn=idn)

    @property
    def default_mount(self):
        return None

    @property
    def default_gripper(self):
        return "Robotiq140Gripper"

    @property
    def default_controller_config(self):
        return "default_iiwa"

    @property
    def init_qpos(self):
        #return np.array([0.000, 0.000, 0.000, 0.35, 0.000, -1.5708, 0.000, -1.9208, -1.570796])
        #return np.array([0.000, -0.500, -7.287e-02,  3.749e-01,-6.545e-02, -1.718e+00,  2.973e-02, -2.092e+00,-1.426e+00])
        return np.array([-7.287e-02,  3.749e-01,-6.545e-02, -1.718e+00,  2.973e-02, -2.092e+00,-1.426e+00])
        #return np.array([0.000, 0.000, 0.000, 0.650, 0.000, -1.890, 0.000, 0.600, 0.000])

        #return np.array([-0.92930329, -0.03923664, 0.86277056, -1.77551591, 0.04806059, -1.80642223, 1.55168235, 0.8, 1.2])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (0, 0, 0),#lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"
