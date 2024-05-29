import argparse

import tensorflow as tf

from garage import wrap_experiment
from garage.experiment import MetaEvaluator
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import RL2PPO
from garage.tf.algos.rl2 import RL2Env, RL2Worker
from garage.tf.policies import GaussianGRUPolicy
from garage.trainer import TFTrainer
from robosuite.garage.ml_robosuite import SawyerMLRobosuite
from robosuite.garage.robosuite_task_sampler import RobosuiteTaskSampler


@wrap_experiment(snapshot_mode='gap', snapshot_gap=10)
def ml_rl2_ppo(ctxt, seed, epochs, episodes_per_task, meta_batch_size):
    """Function which sets up and starts the RL2 based Meta Learning experiment Meta 7 on the Robosuite benchmark.
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
    all_ml_train_subtasks = RobosuiteTaskSampler(ml, 'train', lambda env, _: RL2Env(env))
    tasks = all_ml_train_subtasks.sample(meta_batch_size)
    all_ml_test_subtasks = RobosuiteTaskSampler(ml, 'test', lambda env, _: RL2Env(env))
    env = tasks[0]()
    # sampler_test_subtasks = SetTaskSampler(RobosuiteMLSetTaskEnv, env=RobosuiteMLSetTaskEnv(ml, 'test'))

    with TFTrainer(snapshot_config=ctxt) as trainer:
        policy = GaussianGRUPolicy(name='policy',
                                   hidden_dim=256,
                                   env_spec=env.spec,
                                   hidden_nonlinearity=tf.nn.tanh,
                                   state_include_action=False,
                                   recurrent_nonlinearity=tf.nn.sigmoid)

        baseline = LinearFeatureBaseline(env_spec=env.spec)

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
            envs=tasks,
            max_episode_length=env.spec.max_episode_length,
            is_tf_worker=True,
            n_workers=meta_batch_size,
            worker_class=RL2Worker,
            worker_args=dict(n_episodes_per_trial=episodes_per_task))

        algo = RL2PPO(meta_batch_size=meta_batch_size,
                      task_sampler=all_ml_train_subtasks,
                      env_spec=env.spec,
                      policy=policy,
                      baseline=baseline,
                      sampler=sampler,
                      discount=0.99,
                      gae_lambda=0.95,
                      lr_clip_range=0.2,
                      optimizer_args=dict(batch_size=32,
                                          max_optimization_epochs=10),
                      stop_entropy_gradient=True,
                      entropy_method='max',
                      policy_ent_coeff=5e-6,
                      center_adv=False,
                      meta_evaluator=meta_evaluator,
                      episodes_per_trial=episodes_per_task,
                      n_epochs_per_eval=10)

        trainer.setup(algo, tasks)
        trainer.train(n_epochs=epochs,
                      batch_size=episodes_per_task * env.spec.max_episode_length * meta_batch_size)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='Random seed to use for reproducibility')
    parser.add_argument('--epochs', type=int, default=3500, help='Epochs to execute')
    parser.add_argument('--episodes_per_task', type=int, default=10, help='Number of episodes to sample per task')  # 10 default
    parser.add_argument('--meta_batch_size', type=int, default=7,  # 7 default
                        help='Tasks which are sampled per rollout (=trials in the original RL2 paper)')

    args = parser.parse_args()
    ml_rl2_ppo(seed=args.seed, epochs=args.epochs, episodes_per_task=args.episodes_per_task,
                 meta_batch_size=args.meta_batch_size)
