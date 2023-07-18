import re
from collections import OrderedDict

import numpy as np

from robosuite.garage.robosuite_sawyer_pickplaceall import SawyerPickplaceallRobosuiteEnv
from robosuite.garage.robosuite_sawyer_pickplacebread import SawyerPickplacebreadRobosuiteEnv
from robosuite.garage.robosuite_sawyer_pickplacemilk import SawyerPickplacemilkRobosuiteEnv
from robosuite.garage.robosuite_sawyer_pickplacecereal import SawyerPickplacecerealRobosuiteEnv
from robosuite.garage.robosuite_sawyer_pickplacecan import SawyerPickplacecanRobosuiteEnv
from robosuite.garage.robosuite_sawyer_blocklifting import SawyerBlockliftingRobosuiteEnv

ALL_ROBOSUITE_ENVIRONMENTS = OrderedDict(
    (
        ("blocklifting", SawyerBlockliftingRobosuiteEnv),
        ("pick-place-bread", SawyerPickplacebreadRobosuiteEnv),
        ("pick-place-milk", SawyerPickplacemilkRobosuiteEnv),
        ("pick-place-cereal", SawyerPickplacecerealRobosuiteEnv),
        ("pick-place-can", SawyerPickplacecanRobosuiteEnv),
        ("pick-place-all", SawyerPickplaceallRobosuiteEnv),
    )
)

ROBOSUITEML = OrderedDict(
    (
        (
            "train",
            OrderedDict(
                (
                    ("blocklifting", SawyerBlockliftingRobosuiteEnv),
                    ("pick-place-bread", SawyerPickplacebreadRobosuiteEnv),
                    ("pick-place-milk", SawyerPickplacemilkRobosuiteEnv),
                    ("pick-place-cereal", SawyerPickplacecerealRobosuiteEnv),
                    ("pick-place-can", SawyerPickplacecanRobosuiteEnv),
                )
            ),
        ),
        (
            "test",
            OrderedDict(
                (
                    ("pick-place-all", SawyerPickplaceallRobosuiteEnv),
                )
            ),
        ),
    )
)

robosuiteml_train_args_kwargs = {
    key: dict(
        args=[],
        kwargs={
            "task_id": list(ALL_ROBOSUITE_ENVIRONMENTS.keys()).index(key),
        },
    )
    for key, _ in ROBOSUITEML["train"].items()
}

robosuiteml_test_args_kwargs = {
    key: dict(args=[], kwargs={"task_id": list(ALL_ROBOSUITE_ENVIRONMENTS.keys()).index(key)})
    for key, _ in ROBOSUITEML["test"].items()
}

ML10_ARGS_KWARGS = dict(
    train=robosuiteml_train_args_kwargs_train_args_kwargs,
    test=robosuiteml_test_args_kwargs_test_args_kwargs,
)