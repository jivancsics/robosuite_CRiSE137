"""This file presents a collection of dictionaries which organize all Robosuite environment tasks in the appropriate
combinations for executing the Reinforcement Meta Learning (RML) algorithms MAML and RL2.

There are two common types of basic RML tasks:
- RML on a single task environment (e.g. Blocklifting) called Meta 1. This RML case mimics the ML1 case in Meta-World.
- RML across different task environments which share structure (seven learning tasks and three test tasks in total)
  called Meta 7.
  This RML case mimics the ML10 case in Meta-World.

The two robots Rethink Robotics Sawyer and Kuka IIWA14 (modified in its mounting position to simulate the real-world
circumstances in the TU Vienna ACIN robot lab) operate in the appropriate Robosuite task environments.
"""

from collections import OrderedDict

from robosuite.garage.Robosuite_Sawyer.robosuite_sawyer_pickplacebread import SawyerPickplacebreadRobosuiteEnv
from robosuite.garage.Robosuite_Sawyer.robosuite_sawyer_pickplacemilk import SawyerPickplacemilkRobosuiteEnv
from robosuite.garage.Robosuite_Sawyer.robosuite_sawyer_pickplacecereal import SawyerPickplacecerealRobosuiteEnv
from robosuite.garage.Robosuite_Sawyer.robosuite_sawyer_pickplacecan import SawyerPickplacecanRobosuiteEnv
from robosuite.garage.Robosuite_Sawyer.robosuite_sawyer_blocklifting import SawyerBlockliftingRobosuiteEnv
from robosuite.garage.Robosuite_Sawyer.robosuite_sawyer_door import SawyerDoorRobosuiteEnv
from robosuite.garage.Robosuite_Sawyer.robosuite_sawyer_nutassembly import SawyerNutassemblyRobosuiteEnv
from robosuite.garage.Robosuite_Sawyer.robosuite_sawyer_nutassemblyround import SawyerNutassemblyroundRobosuiteEnv
from robosuite.garage.Robosuite_Sawyer.robosuite_sawyer_nutassemblysquare import SawyerNutassemblysquareRobosuiteEnv
from robosuite.garage.Robosuite_Sawyer.robosuite_sawyer_stack import SawyerStackRobosuiteEnv
from robosuite.garage.Robosuite_Sawyer.robosuite_sawyer_canlifting import SawyerCanliftingRobosuiteEnv
from robosuite.garage.Robosuite_Sawyer.robosuite_sawyer_cereallifting import SawyerCerealliftingRobosuiteEnv
from robosuite.garage.Robosuite_Sawyer.robosuite_sawyer_breadlifting import SawyerBreadliftingRobosuiteEnv
from robosuite.garage.Robosuite_Sawyer.robosuite_sawyer_lemonlifting import SawyerLemonliftingRobosuiteEnv
from robosuite.garage.Robosuite_Sawyer.robosuite_sawyer_blockliftingMeta3 import SawyerBlockliftingMeta3RobosuiteEnv

from robosuite.garage.Robosuite_IIWA14_extended_nolinear.robosuite_IIWA14_extended_nolinear_pickplacebread import (
    IIWA14PickplacebreadRobosuiteEnv)
from robosuite.garage.Robosuite_IIWA14_extended_nolinear.robosuite_IIWA14_extended_nolinear_pickplacemilk import (
    IIWA14PickplacemilkRobosuiteEnv)
from robosuite.garage.Robosuite_IIWA14_extended_nolinear.robosuite_IIWA14_extended_nolinear_pickplacecereal import (
    IIWA14PickplacecerealRobosuiteEnv)
from robosuite.garage.Robosuite_IIWA14_extended_nolinear.robosuite_IIWA14_extended_nolinear_pickplacecan import (
    IIWA14PickplacecanRobosuiteEnv)
from robosuite.garage.Robosuite_IIWA14_extended_nolinear.robosuite_IIWA14_extended_nolinear_blocklifting import (
    IIWA14BlockliftingRobosuiteEnv)
from robosuite.garage.Robosuite_IIWA14_extended_nolinear.robosuite_IIWA14_extended_nolinear_door import (
    IIWA14DoorRobosuiteEnv)
from robosuite.garage.Robosuite_IIWA14_extended_nolinear.robosuite_IIWA14_extended_nolinear_nutassembly import (
    IIWA14NutassemblyRobosuiteEnv)
from robosuite.garage.Robosuite_IIWA14_extended_nolinear.robosuite_IIWA14_extended_nolinear_nutassemblyround import (
    IIWA14NutassemblyroundRobosuiteEnv)
from robosuite.garage.Robosuite_IIWA14_extended_nolinear.robosuite_IIWA14_extended_nolinear_nutassemblysquare import (
    IIWA14NutassemblysquareRobosuiteEnv)
from robosuite.garage.Robosuite_IIWA14_extended_nolinear.robosuite_IIWA14_extended_nolinear_stack import (
    IIWA14StackRobosuiteEnv)
from robosuite.garage.Robosuite_IIWA14_extended_nolinear.robosuite_IIWA14_extended_nolinear_canlifting import (
    IIWA14CanliftingRobosuiteEnv)
from robosuite.garage.Robosuite_IIWA14_extended_nolinear.robosuite_IIWA14_extended_nolinear_cereallifting import (
    IIWA14CerealliftingRobosuiteEnv)
from robosuite.garage.Robosuite_IIWA14_extended_nolinear.robosuite_IIWA14_extended_nolinear_breadlifting import (
    IIWA14BreadliftingRobosuiteEnv)
from robosuite.garage.Robosuite_IIWA14_extended_nolinear.robosuite_IIWA14_extended_nolinear_lemonlifting import (
    IIWA14LemonliftingRobosuiteEnv)
from robosuite.garage.Robosuite_IIWA14_extended_nolinear.robosuite_IIWA14_extended_nolinear_blockliftingMeta3 import (
    IIWA14BlockliftingMeta3RobosuiteEnv)

"""
Exclude wipe task due to the different action space (no gripper used here).
Exclude Pick&Place mixed task due to the high complexity.
---------------------------------------------------------
from robosuite.garage.robosuite_sawyer_wipe import SawyerWipeRobosuiteEnv
from robosuite.garage.robosuite_sawyer_pickplaceall import SawyerPickplaceallRobosuiteEnv
from robosuite.garage.Robosuite_IIWA14_extended_nolinear.robosuite_IIWA14_extended_nolinear_wipe import (
    IIWA14WipeRobosuiteEnv)
from robosuite.garage.Robosuite_IIWA14_extended_nolinear.robosuite_IIWA14_extended_nolinear_pickplaceall import (
    IIWA14PickplaceallRobosuiteEnv)
"""

# Rethink Robotics Sawyer agent

ALL_ROBOSUITE_SAWYER_ENVIRONMENTS = OrderedDict(
    (
        ("blocklifting", SawyerBlockliftingRobosuiteEnv()),
        ("pick-place-bread", SawyerPickplacebreadRobosuiteEnv()),
        ("pick-place-milk", SawyerPickplacemilkRobosuiteEnv()),
        ("pick-place-cereal", SawyerPickplacecerealRobosuiteEnv()),
        ("pick-place-can", SawyerPickplacecanRobosuiteEnv()),
        ("nut-assembly-round", SawyerNutassemblyroundRobosuiteEnv()),
        ("nut-assembly-square", SawyerNutassemblysquareRobosuiteEnv()),
        ("nut-assembly-mixed", SawyerNutassemblyRobosuiteEnv()),
        ("door-open", SawyerDoorRobosuiteEnv()),
        ("stack-blocks", SawyerStackRobosuiteEnv()),
        ("canlifting", SawyerCanliftingRobosuiteEnv()),
        ("cereallifting", SawyerCerealliftingRobosuiteEnv()),
        ("breadlifting", SawyerBreadliftingRobosuiteEnv()),
        ("lemonlifting", SawyerLemonliftingRobosuiteEnv()),
        ("blockliftingMeta3", SawyerBlockliftingMeta3RobosuiteEnv()),

        # Defined but unused environments
        # -------------------------------
        # ("pick-place-mixed", SawyerPickplaceallRobosuiteEnv()),
        # ("wipe-board", SawyerWipeRobosuiteEnv()),
        # """

    )
)

# Meta 1 tasks dict

ALL_ROBOSUITE_SINGLE_ML_TASK_SAWYER_ENVIRONMENTS = OrderedDict(
    (
        ("blocklifting", SawyerBlockliftingRobosuiteEnv(single_task_ml=True)),
        ("pick-place-bread", SawyerPickplacebreadRobosuiteEnv(single_task_ml=True)),
        ("pick-place-milk", SawyerPickplacemilkRobosuiteEnv(single_task_ml=True)),
        ("pick-place-cereal", SawyerPickplacecerealRobosuiteEnv(single_task_ml=True)),
        ("pick-place-can", SawyerPickplacecanRobosuiteEnv(single_task_ml=True)),
        ("nut-assembly-round", SawyerNutassemblyroundRobosuiteEnv(single_task_ml=True)),
        ("nut-assembly-square", SawyerNutassemblysquareRobosuiteEnv(single_task_ml=True)),
        ("nut-assembly-mixed", SawyerNutassemblyRobosuiteEnv(single_task_ml=True)),
        ("door-open", SawyerDoorRobosuiteEnv(single_task_ml=True)),
        ("stack-blocks", SawyerStackRobosuiteEnv(single_task_ml=True)),
        ("canlifting", SawyerCanliftingRobosuiteEnv(single_task_ml=True)),
        ("cereallifting", SawyerCerealliftingRobosuiteEnv(single_task_ml=True)),
        ("breadlifting", SawyerBreadliftingRobosuiteEnv(single_task_ml=True)),
        ("lemonlifting", SawyerLemonliftingRobosuiteEnv(single_task_ml=True)),
        ("blockliftingMeta3", SawyerBlockliftingMeta3RobosuiteEnv(single_task_ml=True)),
    )
)

# Meta 3 tasks dict

ROBOSUITEMETA3_SAWYER = OrderedDict(
    (
        (
            "train",
            OrderedDict(
                (
                    ("blockliftingMeta3", SawyerBlockliftingMeta3RobosuiteEnv(single_task_ml=True)),
                    ("cereallifting", SawyerCerealliftingRobosuiteEnv(single_task_ml=True)),
                    ("lemonlifting", SawyerLemonliftingRobosuiteEnv(single_task_ml=True)),
                )
            ),
        ),
        (
            "test",
            OrderedDict(
                (
                    ("canlifting", SawyerCanliftingRobosuiteEnv(single_task_ml=True)),
                    ("breadlifting", SawyerBreadliftingRobosuiteEnv(single_task_ml=True)),
                )
            ),
        ),
    )
)

# Meta 7 tasks dict

ROBOSUITEML_SAWYER = OrderedDict(
    (
        (
            "train",
            OrderedDict(
                (
                    ("blocklifting", SawyerBlockliftingRobosuiteEnv()),
                    ("pick-place-bread", SawyerPickplacebreadRobosuiteEnv()),
                    ("pick-place-milk", SawyerPickplacemilkRobosuiteEnv()),
                    ("pick-place-cereal", SawyerPickplacecerealRobosuiteEnv()),
                    ("nut-assembly-round", SawyerNutassemblyroundRobosuiteEnv()),
                    ("nut-assembly-mixed", SawyerNutassemblyRobosuiteEnv()),
                    ("door-open", SawyerDoorRobosuiteEnv()),
                )
            ),
        ),
        (
            "test",
            OrderedDict(
                (
                    ("pick-place-can", SawyerPickplacecanRobosuiteEnv()),
                    ("nut-assembly-square", SawyerNutassemblysquareRobosuiteEnv()),
                    ("stack-blocks", SawyerStackRobosuiteEnv()),
                )
            ),
        ),
    )
)

robosuitemeta3_sawyer_train_args_kwargs = {
    key: dict(args=[], kwargs={"task_id": list(ALL_ROBOSUITE_SAWYER_ENVIRONMENTS.keys()).index(key)})
    for key, _ in ROBOSUITEMETA3_SAWYER["train"].items()
}

robosuitemeta3_sawyer_test_args_kwargs = {
    key: dict(args=[], kwargs={"task_id": list(ALL_ROBOSUITE_SAWYER_ENVIRONMENTS.keys()).index(key)})
    for key, _ in ROBOSUITEMETA3_SAWYER["test"].items()
}

Meta3_SAWYER_ARGS_KWARGS = dict(
    train=robosuitemeta3_sawyer_train_args_kwargs,
    test=robosuitemeta3_sawyer_test_args_kwargs,
)

robosuiteml_sawyer_train_args_kwargs = {
    key: dict(args=[], kwargs={"task_id": list(ALL_ROBOSUITE_SAWYER_ENVIRONMENTS.keys()).index(key)})
    for key, _ in ROBOSUITEML_SAWYER["train"].items()
}

robosuiteml_sawyer_test_args_kwargs = {
    key: dict(args=[], kwargs={"task_id": list(ALL_ROBOSUITE_SAWYER_ENVIRONMENTS.keys()).index(key)})
    for key, _ in ROBOSUITEML_SAWYER["test"].items()
}

Meta7_SAWYER_ARGS_KWARGS = dict(
    train=robosuiteml_sawyer_train_args_kwargs,
    test=robosuiteml_sawyer_test_args_kwargs,
)

ROBOSUITESINGLEML_SAWYER = OrderedDict((("train", ALL_ROBOSUITE_SINGLE_ML_TASK_SAWYER_ENVIRONMENTS),
                                        ("test", ALL_ROBOSUITE_SINGLE_ML_TASK_SAWYER_ENVIRONMENTS)))

Meta1_SAWYER_args_kwargs = {
    key: dict(
        args=[],
        kwargs={
            "task_id": list(ALL_ROBOSUITE_SINGLE_ML_TASK_SAWYER_ENVIRONMENTS.keys()).index(key),
        },
    )
    for key, _ in ROBOSUITESINGLEML_SAWYER["train"].items()
}


# Kuka IIWA14_extended_nolinear agent

ALL_ROBOSUITE_IIWA14_ENVIRONMENTS = OrderedDict(
    (
        ("blocklifting", IIWA14BlockliftingRobosuiteEnv()),
        ("pick-place-bread", IIWA14PickplacebreadRobosuiteEnv()),
        ("pick-place-milk", IIWA14PickplacemilkRobosuiteEnv()),
        ("pick-place-cereal", IIWA14PickplacecerealRobosuiteEnv()),
        ("pick-place-can", IIWA14PickplacecanRobosuiteEnv()),
        ("nut-assembly-round", IIWA14NutassemblyroundRobosuiteEnv()),
        ("nut-assembly-square", IIWA14NutassemblysquareRobosuiteEnv()),
        ("nut-assembly-mixed", IIWA14NutassemblyRobosuiteEnv()),
        ("door-open", IIWA14DoorRobosuiteEnv()),
        ("stack-blocks", IIWA14StackRobosuiteEnv()),
        ("canlifting", IIWA14CanliftingRobosuiteEnv()),
        ("cereallifting", IIWA14CerealliftingRobosuiteEnv()),
        ("breadlifting", IIWA14BreadliftingRobosuiteEnv()),
        ("lemonlifting", IIWA14LemonliftingRobosuiteEnv()),
        ("blockliftingMeta3", IIWA14BlockliftingMeta3RobosuiteEnv()),

        # Defined but unused environments
        # -------------------------------
        # ("pick-place-mixed", IIWA14PickplaceallRobosuiteEnv()),
        # ("wipe-board", IIWA14WipeRobosuiteEnv()),
        # """

    )
)

# Meta 1 tasks dict

ALL_ROBOSUITE_SINGLE_ML_TASK_IIWA14_ENVIRONMENTS = OrderedDict(
    (
        ("blocklifting", IIWA14BlockliftingRobosuiteEnv(single_task_ml=True)),
        ("pick-place-bread", IIWA14PickplacebreadRobosuiteEnv(single_task_ml=True)),
        ("pick-place-milk", IIWA14PickplacemilkRobosuiteEnv(single_task_ml=True)),
        ("pick-place-cereal", IIWA14PickplacecerealRobosuiteEnv(single_task_ml=True)),
        ("pick-place-can", IIWA14PickplacecanRobosuiteEnv(single_task_ml=True)),
        ("nut-assembly-round", IIWA14NutassemblyroundRobosuiteEnv(single_task_ml=True)),
        ("nut-assembly-square", IIWA14NutassemblysquareRobosuiteEnv(single_task_ml=True)),
        ("nut-assembly-mixed", IIWA14NutassemblyRobosuiteEnv(single_task_ml=True)),
        ("door-open", IIWA14DoorRobosuiteEnv(single_task_ml=True)),
        ("stack-blocks", IIWA14StackRobosuiteEnv(single_task_ml=True)),
        ("canlifting", IIWA14CanliftingRobosuiteEnv(single_task_ml=True)),
        ("cereallifting", IIWA14CerealliftingRobosuiteEnv(single_task_ml=True)),
        ("breadlifting", IIWA14BreadliftingRobosuiteEnv(single_task_ml=True)),
        ("lemonlifting", IIWA14LemonliftingRobosuiteEnv(single_task_ml=True)),
        ("blockliftingMeta3", IIWA14BlockliftingMeta3RobosuiteEnv(single_task_ml=True)),
    )
)

# Meta 3 tasks dict

ROBOSUITEMETA3_IIWA14 = OrderedDict(
    (
        (
            "train",
            OrderedDict(
                (
                    ("blockliftingMeta3", IIWA14BlockliftingMeta3RobosuiteEnv(single_task_ml=True)),
                    ("cereallifting", IIWA14CerealliftingRobosuiteEnv(single_task_ml=True)),
                    ("lemonlifting", IIWA14LemonliftingRobosuiteEnv(single_task_ml=True)),
                )
            ),
        ),
        (
            "test",
            OrderedDict(
                (
                    ("canlifting", IIWA14CanliftingRobosuiteEnv(single_task_ml=True)),
                    ("breadlifting", IIWA14BreadliftingRobosuiteEnv(single_task_ml=True)),
                )
            ),
        ),
    )
)

# Meta 7 tasks dict

ROBOSUITEML_IIWA14 = OrderedDict(
    (
        (
            "train",
            OrderedDict(
                (
                    ("blocklifting", IIWA14BlockliftingRobosuiteEnv()),
                    ("pick-place-bread", IIWA14PickplacebreadRobosuiteEnv()),
                    ("pick-place-milk", IIWA14PickplacemilkRobosuiteEnv()),
                    ("pick-place-cereal", IIWA14PickplacecerealRobosuiteEnv()),
                    ("nut-assembly-round", IIWA14NutassemblyroundRobosuiteEnv()),
                    ("nut-assembly-mixed", IIWA14NutassemblyRobosuiteEnv()),
                    ("door-open", IIWA14DoorRobosuiteEnv()),
                )
            ),
        ),
        (
            "test",
            OrderedDict(
                (
                    ("pick-place-can", IIWA14PickplacecanRobosuiteEnv()),
                    ("nut-assembly-square", IIWA14NutassemblysquareRobosuiteEnv()),
                    ("stack-blocks", IIWA14StackRobosuiteEnv()),
                )
            ),
        ),
    )
)

robosuitemeta3_iiwa14_train_args_kwargs = {
    key: dict(args=[], kwargs={"task_id": list(ALL_ROBOSUITE_IIWA14_ENVIRONMENTS.keys()).index(key)})
    for key, _ in ROBOSUITEMETA3_IIWA14["train"].items()
}

robosuitemeta3_iiwa14_test_args_kwargs = {
    key: dict(args=[], kwargs={"task_id": list(ALL_ROBOSUITE_IIWA14_ENVIRONMENTS.keys()).index(key)})
    for key, _ in ROBOSUITEMETA3_IIWA14["test"].items()
}

Meta3_IIWA14_ARGS_KWARGS = dict(
    train=robosuitemeta3_iiwa14_train_args_kwargs,
    test=robosuitemeta3_iiwa14_test_args_kwargs,
)

robosuiteml_iiwa14_train_args_kwargs = {
    key: dict(args=[], kwargs={"task_id": list(ALL_ROBOSUITE_IIWA14_ENVIRONMENTS.keys()).index(key)})
    for key, _ in ROBOSUITEML_IIWA14["train"].items()
}

robosuiteml_iiwa14_test_args_kwargs = {
    key: dict(args=[], kwargs={"task_id": list(ALL_ROBOSUITE_IIWA14_ENVIRONMENTS.keys()).index(key)})
    for key, _ in ROBOSUITEML_IIWA14["test"].items()
}

Meta7_IIWA14_ARGS_KWARGS = dict(
    train=robosuiteml_iiwa14_train_args_kwargs,
    test=robosuiteml_iiwa14_test_args_kwargs,
)

ROBOSUITESINGLEML_IIWA14 = OrderedDict((("train", ALL_ROBOSUITE_SINGLE_ML_TASK_IIWA14_ENVIRONMENTS),
                                        ("test", ALL_ROBOSUITE_SINGLE_ML_TASK_IIWA14_ENVIRONMENTS)))

Meta1_IIWA14_args_kwargs = {
    key: dict(
        args=[],
        kwargs={
            "task_id": list(ALL_ROBOSUITE_SINGLE_ML_TASK_IIWA14_ENVIRONMENTS.keys()).index(key),
        },
    )
    for key, _ in ROBOSUITESINGLEML_IIWA14["train"].items()
}
