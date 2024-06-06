"""A callable that "updates" an environment."""
import warnings
from copy import deepcopy


class EnvUpdate:
    """A callable that "updates" an environment.

    Implementors of this interface can be called on environments to update
    them. The passed in environment should then be ignored, and the returned
    one used instead.

    Since no new environment needs to be passed in, this type can also
    be used to construct new environments.

    """

    # pylint: disable=too-few-public-methods

    def __call__(self, old_env=None):
        """Update an environment.

        Note that this implementation does nothing.

        Args:
            old_env (Environment or None): Previous environment. Should not be
                used after being passed in, and should not be closed.

        Returns:
            Environment: The new, updated environment.

        """
        return old_env


class NewEnvUpdate(EnvUpdate):
    """:class:`~EnvUpdate` that creates a new environment every update.

    Args:
        env_constructor (Callable[Environment]): Callable that constructs an
            environment.

    """

    # pylint: disable=too-few-public-methods

    def __init__(self, env_constructor):
        self._env_constructor = env_constructor

    def __call__(self, old_env=None):
        """Update an environment.

        Args:
            old_env (Environment or None): Previous environment. Should not be
                used after being passed in, and should not be closed.

        Returns:
            Environment: The new, updated environment.

        """
        if old_env:
            old_env.close()
        return self._env_constructor()


class SetTaskUpdate(EnvUpdate):
    """:class:`~EnvUpdate` that calls set_task with the provided task.

    Args:
        env_type (type): Type of environment.
        task (object): Opaque task type.
        wrapper_constructor (Callable[garage.Env, garage.Env] or None):
            Callable that wraps constructed environments.

    """

    # pylint: disable=too-few-public-methods

    def __init__(self, env_type, task, wrapper_constructor):
        # Type check for class not available due to the needed special structure inherited
        # through the interaction with Robosuite via suite.make()
        # if not isinstance(env_type, type):
        #     raise ValueError('env_type should be a type, not '
        #                      f'{type(env_type)!r}')
        self._env_type = env_type
        self._task = task
        self._wrapper_cons = wrapper_constructor

    def _make_env(self) -> object:
        """Construct the environment, wrapping if necessary.

        Returns:
            garage.Env: The (possibly wrapped) environment.

        """
        self._env_type.set_task(self._task)
        env = self._env_type()
        if self._wrapper_cons is not None:
            env = self._wrapper_cons(env, self._task)
        return env

    def __call__(self, old_env=None):
        """Update an environment.

        Args:
            old_env (Environment or None): Previous environment. Should not be
                used after being passed in, and should not be closed.

        Returns:
            Environment: The new, updated environment.

        """
        # We need exact type equality, not just a subtype
        # pylint: disable=unidiomatic-typecheck
        # checkenv = deepcopy(self._env_type)
        # checkenv.set_task(self._task)
        # comparison_env = checkenv()
        # if self._wrapper_cons is not None:
        #     env = self._wrapper_cons(comparison_env, self._task)

        if old_env is None:
            return self._make_env()
        #   elif not isinstance(type(getattr(old_env, 'unwrapped', old_env)),
        #                 type(getattr(comparison_env, 'unwrapped'))):
        #  elif type(getattr(old_env, 'unwrapped', old_env)) != type(getattr(comparison_env, 'unwrapped')):
        #     warnings.warn('SetTaskEnvUpdate is closing an environment. This '
        #                   'may indicate a very slow TaskSampler setup.')
        #     old_env.close()
        #     return self._make_env()
        else:
            # Slow sampling setup due to the fact that the task is not settable without deeply changing
            # the underlying class structure of the environments (see robosuite/environments/manipulation)
            #
            # old_env.set_task(self._task)
            # return old_env
            # warnings.warn('SetTaskEnvUpdate is closing an environment. This '
            #               'may indicate a very slow TaskSampler setup.')
            old_env.close()
            return self._make_env()



class ExistingEnvUpdate(EnvUpdate):
    """:class:`~EnvUpdate` that carries an already constructed environment.

    Args:
        env (Environment): The environment.

    """

    def __init__(self, env):
        self._env = env

    def __call__(self, old_env=None):
        """Update an environment.

        This implementation does not close the old environment.

        Args:
            old_env (Environment or None): Previous environment. Should not be
                used after being passed in, and should not be closed.

        Returns:
            Environment: The new, updated environment.

        """
        return self._env

    def __getstate__(self):
        """Get the pickle state.

        Returns:
            dict: The pickled state.

        """
        warnings.warn('ExistingEnvUpdate is generally not the most efficient '
                      'method of transmitting environments to other '
                      'processes.')
        return self.__dict__