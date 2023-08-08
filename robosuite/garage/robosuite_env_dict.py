from collections import OrderedDict

from robosuite.garage.robosuite_sawyer_pickplacebread import SawyerPickplacebreadRobosuiteEnv
from robosuite.garage.robosuite_sawyer_pickplacemilk import SawyerPickplacemilkRobosuiteEnv
from robosuite.garage.robosuite_sawyer_pickplacecereal import SawyerPickplacecerealRobosuiteEnv
from robosuite.garage.robosuite_sawyer_pickplacecan import SawyerPickplacecanRobosuiteEnv
from robosuite.garage.robosuite_sawyer_blocklifting import SawyerBlockliftingRobosuiteEnv
from robosuite.garage.robosuite_sawyer_door import SawyerDoorRobosuiteEnv
from robosuite.garage.robosuite_sawyer_nutassembly import SawyerNutassemblyRobosuiteEnv
from robosuite.garage.robosuite_sawyer_nutassemblyround import SawyerNutassemblyroundRobosuiteEnv
from robosuite.garage.robosuite_sawyer_nutassemblysquare import SawyerNutassemblysquareRobosuiteEnv
from robosuite.garage.robosuite_sawyer_stack import SawyerStackRobosuiteEnv

"""
Exclude wipe task due to the different action space (no gripper used here).
Exclude Pick&Place mixed task due to the high complexity.
---------------------------------------------------------
from robosuite.garage.robosuite_sawyer_wipe import SawyerWipeRobosuiteEnv
from robosuite.garage.robosuite_sawyer_pickplaceall import SawyerPickplaceallRobosuiteEnv
"""

ALL_ROBOSUITE_ENVIRONMENTS = OrderedDict(
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

ALL_ROBOSUITE_SINGLE_ML_TASK_ENVIRONMENTS = OrderedDict(
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

ROBOSUITEML = OrderedDict(
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
                    ("nut-assembly-square", SawyerNutassemblysquareRobosuiteEnv()),
                    ("door-open", SawyerDoorRobosuiteEnv()),
                )
            ),
        ),
        (
            "test",
            OrderedDict(
                (
                    ("pick-place-can", SawyerPickplacecanRobosuiteEnv()),
                    ("nut-assembly-mixed", SawyerNutassemblyRobosuiteEnv()),
                    ("stack-blocks", SawyerStackRobosuiteEnv()),
                )
            ),
        ),
    )
)

robosuiteml_train_args_kwargs = {
    key: dict(args=[], kwargs={"task_id": list(ALL_ROBOSUITE_ENVIRONMENTS.keys()).index(key)})
    for key, _ in ROBOSUITEML["train"].items()
}

robosuiteml_test_args_kwargs = {
    key: dict(args=[], kwargs={"task_id": list(ALL_ROBOSUITE_ENVIRONMENTS.keys()).index(key)})
    for key, _ in ROBOSUITEML["test"].items()
}

ML10_ARGS_KWARGS = dict(
    train=robosuiteml_train_args_kwargs,
    test=robosuiteml_test_args_kwargs,
)

ROBOSUITESINGLEML = OrderedDict((("train", ALL_ROBOSUITE_SINGLE_ML_TASK_ENVIRONMENTS),
                                 ("test", ALL_ROBOSUITE_SINGLE_ML_TASK_ENVIRONMENTS)))

ML1_args_kwargs = {
    key: dict(
        args=[],
        kwargs={
            "task_id": list(ALL_ROBOSUITE_SINGLE_ML_TASK_ENVIRONMENTS.keys()).index(key),
        },
    )
    for key, _ in ROBOSUITESINGLEML["train"].items()
}
