import argparse
from garage import wrap_experiment
from garage.experiment.deterministic import set_seed
import metaworld
from garage.envs import MetaWorldSetTaskEnv
from garage.experiment import MetaWorldTaskSampler, SetTaskSampler, MetaEvaluator

from garage.sampler import RaySampler
from garage.torch.algos import MAMLTRPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer
import torch
from garage.torch import set_gpu_mode

@wrap_experiment(snapshot_mode='last')
def ml10_maml_trpo(ctxt, seed, epochs, episodes_per_task, meta_batch_size):
    """Function which sets up and starts a MAML based experiment on the ML10 MetaWorld benchmark
    Arguments:
        ctxt: Experiment context configuration from the wrap_experiment wrapper, used by Trainer class
        seed: Random seed to use for reproducibility
        epochs: Epochs to execute until termination of the whole meta train/test cycle
        episodes_per_task: Number of episodes to sample per epoch per training task
        meta_batch_size: Tasks which are sampled per batch
    """
    # Set up the environment
    set_seed(seed)
    ml10 = metaworld.ML10()   
    all_ml10_train_subtasks = MetaWorldTaskSampler(ml10, 'train')
    sampled_subtasks = all_ml10_train_subtasks.sample(meta_batch_size)  # 10 subtasks overall = 1 task per class
    env = sampled_subtasks[0]()
    sampler_test_subtasks = SetTaskSampler(MetaWorldSetTaskEnv, env=MetaWorldSetTaskEnv(ml10, 'test'))


    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(128, 128),
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=torch.tanh)

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    meta_evaluator = MetaEvaluator(test_task_sampler=sampler_test_subtasks,
                                   n_test_tasks=int(meta_batch_size/10)*5,
                                   n_exploration_eps=episodes_per_task,)

    sampler = RaySampler(agents=policy,
                         envs=env,
                         max_episode_length=env.spec.max_episode_length)
                         #n_workers=meta_batch_size)

    # n_workers = how many tasks should be sampled in parallel
    # initializes a worker factory and the worker factory processes
    # self._envs to a list with length n_workers, e.g. if one passes
    # just one env & n_workers = 3 --> [env, env, env] -->
    # usage of that list in start_worker() -->
    # Guess that this list is updated at runtime through
    # self._update_workers(agent_update, env_update) in obtain_samples
    # (line 176, ray_sampler.py), therefore no influence on the result
    # if one passes just a single task environment or a whole list of
    # different task environments (len(env_list)==n_workers)! The
    # task environments themselfs are sampled at runtime and allocated
    # to ray-workers after that to obtain the episodes, one sampled
    # task environment after the other. GUESS: Some ray-workers will
    # stay in idle in the specific MAML-case here as we are initializing
    # num=meta_batch_size workers, but just need num=episodes per task
    # workers


    trainer = Trainer(ctxt)
    algo = MAMLTRPO(env=env,
                    policy=policy,
                    sampler=sampler,
                    task_sampler=all_ml10_train_subtasks,
                    value_function=value_function,
                    meta_batch_size=meta_batch_size,
                    discount=0.99,
                    gae_lambda=1.,
                    inner_lr=1e-4,
                    outer_lr=1e-3,
                    max_kl_step=1e-2,
                    num_grad_updates=1,
                    policy_ent_coeff=0.0,   # 5e-5 mentioned in the MetaWorld paper, but exact type is missing
                    meta_evaluator=meta_evaluator,  # meta_batch_size = how many tasks to sample for training
                    evaluate_every_n_epochs=50)

    # if torch.cuda.is_available():
    #     set_gpu_mode(True)
    # else:
    #     set_gpu_mode(False)
    # algo.to() no GPU mode available for MAMLTRPO
    trainer.setup(algo, env)
    trainer.train(n_epochs=epochs,
                  batch_size=episodes_per_task * env.spec.max_episode_length)   # batch_size = batch length per task




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='Random seed to use for reproducibility')
    parser.add_argument('--epochs', type=int, default=3500, help='Epochs to execute')
    parser.add_argument('--episodes_per_task', type=int, default=10, help='Number of episodes to sample per task')
    parser.add_argument('--meta_batch_size', type=int, default=20,
                        help='Tasks which are sampled per batch')

    args = parser.parse_args()
    ml10_maml_trpo(seed=args.seed, epochs=args.epochs, episodes_per_task=args.episodes_per_task,
                   meta_batch_size=args.meta_batch_size)
