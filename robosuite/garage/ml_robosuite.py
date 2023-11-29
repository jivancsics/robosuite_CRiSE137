import abc
from collections import OrderedDict
from typing import List, NamedTuple, Type
import pickle
import robosuite.garage.robosuite_env_dict as _env_dict
import numpy as np

EnvName = str


class Task(NamedTuple):
    """All data necessary to describe a single MDP.

    Should be passed into a MetaWorldEnv's set_task method.
    """

    env_name: EnvName
    data: bytes  # Contains env parameters like random_init and *a* goal


class Benchmark(abc.ABC):
    """A Benchmark.

    When used to evaluate an algorithm, only a single instance should be used.
    """

    @abc.abstractmethod
    def __init__(self):
        pass

    @property
    def train_classes(self) -> "OrderedDict[EnvName, Type]":
        """Get all of the environment classes used for training."""
        return self._train_classes

    @property
    def test_classes(self) -> "OrderedDict[EnvName, Type]":
        """Get all of the environment classes used for testing."""
        return self._test_classes

    @property
    def train_tasks(self) -> List[Task]:
        """Get all of the training tasks for this benchmark."""
        return self._train_tasks

    @property
    def test_tasks(self) -> List[Task]:
        """Get all of the test tasks for this benchmark."""
        return self._test_tasks


def _encode_task(env_name, data):
    return Task(env_name=env_name, data=pickle.dumps(data))


def _make_tasks(classes, args_kwargs, seed=None, single_task_ml=False):
    if seed is not None:
        st0 = np.random.get_state()
        np.random.seed(seed)
    _N_GOALS = 50
    if single_task_ml:
        _N_GOALS = 100
    tasks = []
    for env_name, args in args_kwargs.items():
        assert len(args["args"]) == 0
        env_cls = classes[env_name]
        env_cls._freeze_rand_vec = False
        env_cls._set_task_called = True
        rand_vecs = []
        kwargs = args["kwargs"].copy()
        del kwargs["task_id"]
        for _ in range(_N_GOALS):
            if env_name == "nut-assembly-mixed":
                y_pos_square = np.random.uniform(low=0.12, high=0.225)
                x_pos_square = np.random.uniform(low=-0.13, high=-0.10)
                rot_square = np.random.uniform(low=0, high=2.0 * np.pi)
                y_pos_round = np.random.uniform(low=-0.225, high=-0.12)
                x_pos_round = np.random.uniform(low=-0.13, high=-0.10)
                rot_round = np.random.uniform(low=0, high=2.0 * np.pi)
                rand_vecs.append([x_pos_square, y_pos_square, rot_square, x_pos_round, y_pos_round, rot_round])
            elif env_name == "nut-assembly-square":
                y_pos_square = np.random.uniform(low=0.10, high=0.225)
                x_pos_square = np.random.uniform(low=-0.13, high=-0.10)
                rot_square = np.random.uniform(low=0, high=np.pi / 4.0)
                rand_vecs.append([x_pos_square, y_pos_square, rot_square])
            elif env_name == "nut-assembly-round":
                y_pos_round = np.random.uniform(low=-0.225, high=-0.10)
                x_pos_round = np.random.uniform(low=-0.13, high=-0.10)
                rot_round = np.random.uniform(low=0, high=np.pi / 2.0)
                rand_vecs.append([x_pos_round, y_pos_round, rot_round])
            elif env_name in ("blocklifting", "canlifting"):
                x_pos = np.random.uniform(low=-0.30, high=0.30)
                y_pos = np.random.uniform(low=-0.30, high=0.30)
                rot = np.random.uniform(low=0, high=np.pi / 4.0)
                rand_vecs.append([x_pos, y_pos, rot])
            elif env_name in ("cereallifting", "breadlifting", "lemonlifting"):
                x_pos = np.random.uniform(low=-0.30, high=0.30)
                y_pos = np.random.uniform(low=-0.30, high=0.30)
                rot = np.random.uniform(low=0, high=np.pi)
                rand_vecs.append([x_pos, y_pos, rot])
            elif env_name == "stack-blocks":
                x_pos_a = np.random.uniform(low=-0.30, high=0.30)
                y_pos_a = np.random.uniform(low=-0.30, high=0.30)
                rot_a = np.random.uniform(low=0, high=np.pi / 4.0)
                vec_cubea = np.array([x_pos_a, y_pos_a])
                while True:
                    x_pos_b = np.random.uniform(low=-0.30, high=0.30)
                    y_pos_b = np.random.uniform(low=-0.30, high=0.30)
                    vec_cubeb = np.array([x_pos_b, y_pos_b])
                    dist_ab = np.linalg.norm(vec_cubea - vec_cubeb)

                    """
                    Cube dimensions cubeA = [0.04, 0.04, 0.04], cubeB = [0.05, 0.05, 0.05] =>
                    radius from origin around cubeA = sqrt(2 * (0.02 * 0.02), 
                    radius cubeB = sqrt(2 * (0.025 * 0.025) => 
                    dist_AB > radius cubeA + radius cubeB, else resampling of cubeB
                    """

                    if dist_ab > (np.sqrt(2 * (0.02 * 0.02)) + np.sqrt(2 * (0.025 * 0.025))):
                        break
                rot_b = np.random.uniform(low=0, high=np.pi / 4.0)
                rand_vecs.append([x_pos_a, y_pos_a, rot_a, x_pos_b, y_pos_b, rot_b])
            elif env_name in ("pick-place-mixed", "pick-place-bread", "pick-place-milk",
                              "pick-place-cereal", "pick-place-can"):
                bin1_x_pos = np.random.uniform(low=-0.1, high=0.1)
                bin2_x_pos = np.random.uniform(low=-0.1, high=0.1)
                bin1_y_pos = np.random.uniform(low=-0.27, high=-0.25)
                bin2_y_pos = np.random.uniform(low=0.28, high=0.3)
                bin12_z_pos = 0.8    # equal table height
                rand_vecs.append([bin1_x_pos, bin1_y_pos, bin2_x_pos, bin2_y_pos, bin12_z_pos])
            elif env_name == "door-open":
                x_pos = np.random.uniform(low=0.07, high=0.09)
                y_pos = np.random.uniform(low=-0.01, high=0.01)
                rot = np.random.uniform(low=-np.pi / 2.0 - 0.25, high=-np.pi / 2.0)
                rand_vecs.append([x_pos, y_pos, rot])
            elif env_name == "wipe-board":
                path_list = []
                for i in range(env_cls.num_markers):
                    if i == 0:  # start position
                        path_list.append(np.random.uniform(-0.4 * 0.7 + 0.01, 0.4 * 0.7 - 0.01))
                        path_list.append(np.random.uniform(-0.4 * 0.7 + 0.01, 0.4 * 0.7 - 0.01))
                        path_list.append(np.random.uniform(-np.pi, np.pi))
                    else:   # rest of the path
                        if np.random.uniform(0, 1) > 0.7:
                            direction = path_list[i * 3 - 1] + np.random.normal(0, 0.5)
                        else:
                            direction = path_list[i * 3 - 1]

                        posnew0 = path_list[i * 3 - 3] + 0.005 * np.sin(direction)
                        posnew1 = path_list[i * 3 - 2] + 0.005 * np.cos(direction)

                        # We keep resampling until we get a valid new position that's on the table
                        while (
                                abs(posnew0) >= 0.4 * 0.7 + 0.01
                                or abs(posnew1) >= 0.4 * 0.7 + 0.01
                        ):
                            direction += np.random.normal(0, 0.5)
                            posnew0 = path_list[i * 3 - 3] + 0.005 * np.sin(direction)
                            posnew1 = path_list[i * 3 - 2] + 0.005 * np.cos(direction)

                        # Append this newly sampled position
                        path_list.append(posnew0)
                        path_list.append(posnew1)
                        path_list.append(direction)

                # Delete direction elements (only necessary for position computation)
                del path_list[2::3]
                # Append the whole path to the rand_vecs list
                rand_vecs.append(path_list)

        unique_task_rand_vecs = np.unique(np.array(rand_vecs), axis=0)
        assert unique_task_rand_vecs.shape[0] == _N_GOALS

        for rand_vec in rand_vecs:
            kwargs = args["kwargs"].copy()
            del kwargs["task_id"]
            kwargs.update(dict(rand_vec=rand_vec, env_cls=env_cls))
            tasks.append(_encode_task(env_name, kwargs))
    if seed is not None:
        np.random.set_state(st0)
    return tasks


def _sawyer_singleml_env_names():
    tasks = list(_env_dict.ROBOSUITESINGLEML_SAWYER["train"])
    return tasks

def _iiwa14_singleml_env_names():
    tasks = list(_env_dict.ROBOSUITESINGLEML_IIWA14["train"])
    return tasks


# Set up the Meta 7 experiment on the Sawyer robot

class SawyerMLRobosuite(Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.ROBOSUITEML_SAWYER["train"]
        self._test_classes = _env_dict.ROBOSUITEML_SAWYER["test"]
        train_kwargs = _env_dict.robosuiteml_sawyer_train_args_kwargs
        self._train_tasks = _make_tasks(
            self._train_classes, train_kwargs, seed=seed
        )
        test_kwargs = _env_dict.robosuiteml_sawyer_test_args_kwargs
        self._test_tasks = _make_tasks(
            self._test_classes, test_kwargs, seed=seed
        )

# Set up the Meta 3 experiment on the Sawyer robot

class SawyerMeta3Robosuite(Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.ROBOSUITEMETA3_SAWYER["train"]
        self._test_classes = _env_dict.ROBOSUITEMETA3_SAWYER["test"]
        train_kwargs = _env_dict.robosuitemeta3_sawyer_train_args_kwargs
        self._train_tasks = _make_tasks(
            self._train_classes, train_kwargs, seed=seed
        )
        test_kwargs = _env_dict.robosuitemeta3_sawyer_test_args_kwargs
        self._test_tasks = _make_tasks(
            self._test_classes, test_kwargs, seed=seed
        )


# Set up the appropriate Meta 1 experiment on the Sawyer robot

class SawyerSingleMLRobosuite(Benchmark):
    ENV_NAMES = _sawyer_singleml_env_names()

    def __init__(self, env_name, seed=None):
        super().__init__()
        if env_name not in _env_dict.ALL_ROBOSUITE_SINGLE_ML_TASK_SAWYER_ENVIRONMENTS:
            raise ValueError(f"{env_name} is not a Robosuite Meta1 environment")
        cls = _env_dict.ALL_ROBOSUITE_SINGLE_ML_TASK_SAWYER_ENVIRONMENTS[env_name]
        self._train_classes = OrderedDict([(env_name, cls)])
        self._test_classes = self._train_classes
        self._train_ = OrderedDict([(env_name, cls)])
        args_kwargs = _env_dict.Meta1_SAWYER_args_kwargs[env_name]

        # Make sure that train tasks and test tasks are not the same
        # Use the built in functionality of _make_tasks to fulfill this requirement
        self._train_tasks = _make_tasks(
            self._train_classes, {env_name: args_kwargs}, seed=seed, single_task_ml=True,)
        self._test_tasks = self._train_tasks[50:100]
        del self._train_tasks[50:100]


# Set up the Meta 7 experiment on the IIWA14 robot

class IIWA14MLRobosuite(Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.ROBOSUITEML_IIWA14["train"]
        self._test_classes = _env_dict.ROBOSUITEML_IIWA14["test"]
        train_kwargs = _env_dict.robosuiteml_iiwa14_train_args_kwargs
        self._train_tasks = _make_tasks(
            self._train_classes, train_kwargs, seed=seed
        )
        test_kwargs = _env_dict.robosuiteml_iiwa14_test_args_kwargs
        self._test_tasks = _make_tasks(
            self._test_classes, test_kwargs, seed=seed
        )


# Set up the Meta 3 experiment on the IIWA14 robot

class IIWA14Meta3Robosuite(Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.ROBOSUITEMETA3_IIWA14["train"]
        self._test_classes = _env_dict.ROBOSUITEMETA3_IIWA14["test"]
        train_kwargs = _env_dict.robosuitemeta3_iiwa14_train_args_kwargs
        self._train_tasks = _make_tasks(
            self._train_classes, train_kwargs, seed=seed
        )
        test_kwargs = _env_dict.robosuitemeta3_iiwa14_test_args_kwargs
        self._test_tasks = _make_tasks(
            self._test_classes, test_kwargs, seed=seed
        )


# Set up the appropriate Meta 1 experiment on the IIWA14 robot

class IIWA14SingleMLRobosuite(Benchmark):
    ENV_NAMES = _iiwa14_singleml_env_names()

    def __init__(self, env_name, seed=None):
        super().__init__()
        if env_name not in _env_dict.ALL_ROBOSUITE_SINGLE_ML_TASK_IIWA14_ENVIRONMENTS:
            raise ValueError(f"{env_name} is not a Robosuite Meta1 environment")
        cls = _env_dict.ALL_ROBOSUITE_SINGLE_ML_TASK_IIWA14_ENVIRONMENTS[env_name]
        self._train_classes = OrderedDict([(env_name, cls)])
        self._test_classes = self._train_classes
        self._train_ = OrderedDict([(env_name, cls)])
        args_kwargs = _env_dict.Meta1_IIWA14_args_kwargs[env_name]

        # Make sure that train tasks and test tasks are not the same
        # Use the built in functionality of _make_tasks to fulfill this requirement
        self._train_tasks = _make_tasks(
            self._train_classes, {env_name: args_kwargs}, seed=seed, single_task_ml=True,)
        self._test_tasks = self._train_tasks[50:100]
        del self._train_tasks[50:100]


__all__ = ["SawyerMLRobosuite", "SawyerMeta3Robosuite", "SawyerSingleMLRobosuite", "IIWA14MLRobosuite",
           "IIWA14Meta3Robosuite", "IIWA14SingleMLRobosuite"]
