import numpy as np

from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import array_to_string, string_to_array, xml_path_completion


class FrameArena(Arena):
    """
    Workspace that contains an empty table.


    Args:
        table_full_size (3-tuple): (L,W,H) full dimensions of the table
        table_friction (3-tuple): (sliding, torsional, rolling) friction parameters of the table
        table_offset (3-tuple): (x,y,z) offset from center of arena when placing table.
            Note that the z value sets the upper limit of the table
        has_legs (bool): whether the table has legs or not
        xml (str): xml file to load arena
    """

    def __init__(
        self,
        #table_full_size=(0.8, 0.8, 0.05),
        #table_friction=(1, 0.005, 0.0001),
        #table_offset=(0, 0, 0.8),
        #has_legs=True,
        xml="arenas/frame_arena.xml",
    ):
        super().__init__(xml_path_completion(xml))

        #self.table_full_size = np.array(table_full_size)
        #self.table_half_size = self.table_full_size / 2
        #self.table_friction = table_friction
        #self.table_offset = table_offset
        #self.center_pos = self.bottom_pos + np.array([0, 0, -self.table_half_size[2]]) + self.table_offset

        #self.table_body = self.worldbody.find("./body[@name='table']")
        #self.table_collision = self.table_body.find("./geom[@name='table_collision']")
        #self.table_visual = self.table_body.find("./geom[@name='table_visual']")
        #self.table_top = self.table_body.find("./site[@name='table_top']")

        self.frame_body = self.worldbody.find("./body[@name='frame']")
        #self.frame_visual = self.frame_body.find("./geom[@name='frame_visual']")
        

        self.configure_location()

    def configure_location(self):
        """Configures correct locations for this arena"""
        self.floor.set("pos", array_to_string(self.bottom_pos))

        #self.frame_body.set("pos", np.array([0, 0, 0]))
        #self.table_collision.set("size", array_to_string(self.table_half_size))
        #self.table_collision.set("friction", array_to_string(self.table_friction))
        #self.table_visual.set("size", array_to_string(self.table_half_size))

        #self.table_top.set("pos", array_to_string(np.array([0, 0, self.table_half_size[2]])))
