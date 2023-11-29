import argparse
from garage import wrap_experiment
from garage.experiment.deterministic import set_seed
from robosuite.garage.robosuiteml_set_task_env import RobosuiteMLSetTaskEnv
from garage.experiment import MetaEvaluator
from robosuite.garage.robosuite_task_sampler import RobosuiteTaskSampler, SetTaskSampler
from robosuite.garage.ml_robosuite import SawyerMLRobosuite
from garage.sampler import RaySampler, LocalSampler
from garage.torch.algos import MAMLTRPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer
import torch
from garage.torch import set_gpu_mode

@wrap_experiment(snapshot_mode='gap', snapshot_gap=5)
def ml_maml_trpo(ctxt, seed, epochs, episodes_per_task, meta_batch_size):
    """Function which sets up and starts the MAML based Meta Learning experiment Meta 7 on the Robosuite benchmark.
    This experiment resembles the ML10 experiment in MetaWorld. Robot used: Rethink Robotics Sawyer.

    Arguments:
        ctxt: Experiment context configuration from the wrap_experiment wrapper, used by Trainer class
        seed: Random seed to use for reproducibility
        epochs: Epochs to execute until termination of the whole meta train/test cycle
        episodes_per_task: Number of episodes to sample per epoch per training task
        meta_batch_size: Tasks which are sampled per batch
    """
    # Set up the environment
    set_seed(seed)
    ml = SawyerMLRobosuite()
    all_ml_train_subtasks = RobosuiteTaskSampler(ml, 'train')
    sampled_subtasks = all_ml_train_subtasks.sample(meta_batch_size)  # 14 subtasks overall = 2 task per class
    all_ml_test_subtasks = RobosuiteTaskSampler(ml, 'test')
    env = sampled_subtasks[0]()
    # sampler_test_subtasks = SetTaskSampler(RobosuiteMLSetTaskEnv, env=RobosuiteMLSetTaskEnv(ml, 'test'))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(128, 128),
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=torch.tanh)

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    meta_evaluator = MetaEvaluator(test_task_sampler=all_ml_test_subtasks,
                                   n_test_tasks=(int(meta_batch_size/7)*len(all_ml_test_subtasks._classes)),
                                   n_exploration_eps=episodes_per_task,
                                   is_robosuite_ml=True,)

    # sampler = RaySampler(agents=policy,
    #                      envs=env,
    #                      max_episode_length=env.spec.max_episode_length)
    #                      #n_workers=meta_batch_size)


    sampler = LocalSampler(
        agents=policy,
        envs=sampled_subtasks,
        max_episode_length=env.spec.max_episode_length,
        is_tf_worker=False,
        n_workers=meta_batch_size,)

    trainer = Trainer(ctxt)
    algo = MAMLTRPO(env=env,
                    policy=policy,
                    sampler=sampler,
                    task_sampler=all_ml_train_subtasks,
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
                    evaluate_every_n_epochs=5)

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
    parser.add_argument('--episodes_per_task', type=int, default=10, help='Number of episodes to sample per task')  # 10 default
    parser.add_argument('--meta_batch_size', type=int, default=14,  # 14 default
                        help='Tasks which are sampled per batch')

    args = parser.parse_args()
    ml_maml_trpo(seed=args.seed, epochs=args.epochs, episodes_per_task=args.episodes_per_task,
                 meta_batch_size=args.meta_batch_size)
