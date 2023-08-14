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

        # Defined but unused environments
        # -------------------------------
        # ("pick-place-mixed", SawyerPickplaceallRobosuiteEnv()),
        # ("wipe-board", SawyerWipeRobosuiteEnv()),
        # """

    )
)

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
    )
)

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

robosuiteml_sawyer_train_args_kwargs = {
    key: dict(args=[], kwargs={"task_id": list(ALL_ROBOSUITE_SAWYER_ENVIRONMENTS.keys()).index(key)})
    for key, _ in ROBOSUITEML_SAWYER["train"].items()
}

robosuiteml_sawyer_test_args_kwargs = {
    key: dict(args=[], kwargs={"task_id": list(ALL_ROBOSUITE_SAWYER_ENVIRONMENTS.keys()).index(key)})
    for key, _ in ROBOSUITEML_SAWYER["test"].items()
}

ML10_SAWYER_ARGS_KWARGS = dict(
    train=robosuiteml_sawyer_train_args_kwargs,
    test=robosuiteml_sawyer_test_args_kwargs,
)

ROBOSUITESINGLEML_SAWYER = OrderedDict((("train", ALL_ROBOSUITE_SINGLE_ML_TASK_SAWYER_ENVIRONMENTS),
                                        ("test", ALL_ROBOSUITE_SINGLE_ML_TASK_SAWYER_ENVIRONMENTS)))

ML1_SAWYER_args_kwargs = {
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

        # Defined but unused environments
        # -------------------------------
        # ("pick-place-mixed", IIWA14PickplaceallRobosuiteEnv()),
        # ("wipe-board", IIWA14WipeRobosuiteEnv()),
        # """

    )
)

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
    )
)

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

robosuiteml_iiwa14_train_args_kwargs = {
    key: dict(args=[], kwargs={"task_id": list(ALL_ROBOSUITE_IIWA14_ENVIRONMENTS.keys()).index(key)})
    for key, _ in ROBOSUITEML_IIWA14["train"].items()
}

robosuiteml_iiwa14_test_args_kwargs = {
    key: dict(args=[], kwargs={"task_id": list(ALL_ROBOSUITE_IIWA14_ENVIRONMENTS.keys()).index(key)})
    for key, _ in ROBOSUITEML_IIWA14["test"].items()
}

ML10_IIWA14_ARGS_KWARGS = dict(
    train=robosuiteml_iiwa14_train_args_kwargs,
    test=robosuiteml_iiwa14_test_args_kwargs,
)

ROBOSUITESINGLEML_IIWA14 = OrderedDict((("train", ALL_ROBOSUITE_SINGLE_ML_TASK_IIWA14_ENVIRONMENTS),
                                        ("test", ALL_ROBOSUITE_SINGLE_ML_TASK_IIWA14_ENVIRONMENTS)))

ML1_IIWA14_args_kwargs = {
    key: dict(
        args=[],
        kwargs={
            "task_id": list(ALL_ROBOSUITE_SINGLE_ML_TASK_IIWA14_ENVIRONMENTS.keys()).index(key),
        },
    )
    for key, _ in ROBOSUITESINGLEML_IIWA14["train"].items()
}
