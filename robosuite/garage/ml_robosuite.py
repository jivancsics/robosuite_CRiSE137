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
        env = env_cls()
        env._freeze_rand_vec = False
        env._set_task_called = True
        rand_vecs = []
        kwargs = args["kwargs"].copy()
        del kwargs["task_id"]
#         env._set_task_inner(**kwargs)
        for _ in range(_N_GOALS):
            # env.reset()
            rand_vecs.append(env._last_rand_vec)
        unique_task_rand_vecs = np.unique(np.array(rand_vecs), axis=0)
        assert unique_task_rand_vecs.shape[0] == _N_GOALS

        env.close()
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
