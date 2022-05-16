Add the line
  from robosuite.environments.manipulation.kuka_linear_env import KukaLinearEnv
to robosuite/__init__.py

Add kuka_linear_env.py to robosuite/environments/manipulation

Add frame_arene.py to robosuite/models/arenas

Add iiwa_extended_robot.py to robosuite/models/robots/manipulators

Add the line
  from .iiwa_extended_robot import IIWA_extended
to robosuite/models/robots/manipulators/__init__.py

extract kuka_linear_arena_meshes.zip to robosuite/models/assets/arenas/meshes

extract iiwa_extended.zip to robosuite/models/assets/robots
