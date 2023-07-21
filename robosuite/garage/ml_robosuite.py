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


_N_GOALS = 50


def _encode_task(env_name, data):
    return Task(env_name=env_name, data=pickle.dumps(data))


def _make_tasks(classes, args_kwargs, seed=None):
    if seed is not None:
        st0 = np.random.get_state()
        np.random.seed(seed)
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
            if env_name is "nut-assembly-mixed":
                y_pos_square = np.random.uniform(low=0.10, high=0.225)
                x_pos_square = np.random.uniform(low=-0.13, high=-0.10)
                rot_square = np.random.uniform(low=0, high=np.pi/2.0)
                y_pos_round = np.random.uniform(low=-0.225, high=-0.10)
                x_pos_round = np.random.uniform(low=-0.13, high=-0.10)
                rot_round = np.random.uniform(low=0, high=np.pi/2.0)
                rand_vecs.append([x_pos_square, y_pos_square, rot_square, x_pos_round, y_pos_round, rot_round])
            elif env_name is "nut-assembly-square":
                y_pos_square = np.random.uniform(low=0.10, high=0.225)
                x_pos_square = np.random.uniform(low=-0.13, high=-0.10)
                rot_square = np.random.uniform(low=0, high=np.pi/2.0)
                rand_vecs.append([x_pos_square, y_pos_square, rot_square])
            elif env_name is "nut-assembly-round":
                y_pos_round = np.random.uniform(low=-0.225, high=-0.10)
                x_pos_round = np.random.uniform(low=-0.13, high=-0.10)
                rot_round = np.random.uniform(low=0, high=np.pi/2.0)
                rand_vecs.append([x_pos_round, y_pos_round, rot_round])
            elif env_name is "blocklifting":
                x_pos = np.random.uniform(low=-0.2, high=0.2)
                y_pos = np.random.uniform(low=-0.2, high=0.2)
                rot = np.random.uniform(low=0, high=np.pi/2.0)
                rand_vecs.append([x_pos, y_pos, rot])
            elif env_name is "stack-blocks":
                x_pos_A = np.random.uniform(low=-0.2, high=0.2)
                y_pos_A = np.random.uniform(low=-0.2, high=0.2)
                rot_A = np.random.uniform(low=0, high=np.pi/2.0)
                x_pos_B = np.random.uniform(low=-0.2, high=0.2)
                y_pos_B = np.random.uniform(low=-0.2, high=0.2)
                rot_B = np.random.uniform(low=0, high=np.pi/2.0)
                rand_vecs.append([x_pos_A, y_pos_A, rot_A, x_pos_B, y_pos_B, rot_B])
            elif env_name in ("pick-place-mixed", "pick-place-bread", "pick-place-milk",
                              "pick-place-cereal", "pick-place-can"):
                bin1_x_pos = np.random.uniform(low=-0.1, high=0.1)
                bin2_x_pos = np.random.uniform(low=-0.1, high=0.1)
                bin1_y_pos = np.random.uniform(low=-0.27, high=-0.25)
                bin2_y_pos = np.random.uniform(low=0.28, high=0.3)
                bin12_z_pos = 0.8    # equal table height
                rand_vecs.append([bin1_x_pos, bin1_y_pos, bin2_x_pos, bin2_y_pos, bin12_z_pos])
            elif env_name is "door-open":
                x_pos = np.random.uniform(low=0.07, high=0.09)
                y_pos = np.random.uniform(low=-0.01, high=0.01)
                rot = np.random.uniform(low=-np.pi/2.0-0.25, high=-np.pi/2.0)
                rand_vecs.append([x_pos, y_pos, rot])
            elif env_name is "wipe-board":
                # TODO:
                # - Continue here: learn how to modify the wipe line! (robosuite/environments/manipulation/wipe.py)
                # - Modify robosuite/wrappers/gym_wrapper.py GymWrapper self.env.spec (line 53) to comply with Garage!
                # - Step through the robosuite_maml example
                pass

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


class MLRobosuite(Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.ROBOSUITEML["train"]
        self._test_classes = _env_dict.ROBOSUITEML["test"]
        train_kwargs = _env_dict.robosuiteml_train_args_kwargs
        self._train_tasks = _make_tasks(
            self._train_classes, train_kwargs, seed=seed
        )
        test_kwargs = _env_dict.robosuiteml_test_args_kwargs
        self._test_tasks = _make_tasks(
            self._test_classes, test_kwargs, seed=seed
        )


__all__ = ["MLRobosuite"]
