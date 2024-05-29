import argparse
from garage import wrap_experiment
from garage.experiment.deterministic import set_seed
import metaworld
from garage.envs import MetaWorldSetTaskEnv
from garage.experiment import MetaWorldTaskSampler, SetTaskSampler, MetaEvaluator

from garage.sampler import LocalSampler, RaySampler
from garage.tf.algos import RL2PPO
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.policies import GaussianGRUPolicy
from garage.trainer import TFTrainer
from garage.tf.algos.rl2 import RL2Env, RL2Worker
import tensorflow as tf


@wrap_experiment(snapshot_mode='gap', snapshot_gap=20)
def ml10_rl2_ppo(ctxt, seed, epochs, episodes_per_task, meta_batch_size):
    """Function which sets up and starts a RL2 based experiment on the ML10 MetaWorld benchmark
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
    all_ml10_train_subtasks = MetaWorldTaskSampler(ml10, 'train', lambda env, _: RL2Env(env))
    sampled_subtasks = all_ml10_train_subtasks.sample(meta_batch_size)
    env = sampled_subtasks[0]()
    sampler_test_subtasks = SetTaskSampler(MetaWorldSetTaskEnv, env=MetaWorldSetTaskEnv(ml10, 'test'),
                                           wrapper=lambda env, _: RL2Env(env))

    with TFTrainer(snapshot_config=ctxt) as trainer:
        policy = GaussianGRUPolicy(name='policy',
                                   hidden_dim=256,
                                   env_spec=env.spec,
                                   hidden_nonlinearity=tf.nn.tanh,
                                   state_include_action=False,
                                   recurrent_nonlinearity=tf.nn.sigmoid)
        # In the RL2 paper, the tuple (s,a,r,d) is the vanilla input of the policy and gets embedded by an embedder
        # phi(s,a,r,d) in the first place. This embedding is fed into Gated Recurrent Units (GRUs because of the
        # vanishing and exploding gradients problems with simple RNN types) and after that, fed into a fully connected.
        # Output activation depends on the action space needed (categorical/discrete-->softmax, continous-->dense layer)

        meta_evaluator = MetaEvaluator(test_task_sampler=sampler_test_subtasks,
                                       n_test_tasks=int(meta_batch_size/10)*5,
                                       n_exploration_eps=episodes_per_task,)

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        sampler = LocalSampler(
            agents=policy,
            envs=sampled_subtasks,
            max_episode_length=env.spec.max_episode_length,
            is_tf_worker=True,
            n_workers=meta_batch_size,
            worker_class=RL2Worker,
            worker_args=dict(n_episodes_per_trial=episodes_per_task))

        # sampler = RaySampler(
        #     agents=policy,
        #     envs=envs,
        #     max_episode_length=env.spec.max_episode_length,
        #     is_tf_worker=True,
        #     n_workers=meta_batch_size,
        #     worker_class=RL2Worker,
        #     worker_args=dict(n_episodes_per_trial=episodes_per_task))

        algo = RL2PPO(meta_batch_size=meta_batch_size,
                      task_sampler=all_ml10_train_subtasks,
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
                      n_epochs_per_eval=50)

        trainer.setup(algo, sampled_subtasks)

        trainer.train(n_epochs=epochs,
                      batch_size=episodes_per_task *                                # ep_per_task = ep per trial
                                 env.spec.max_episode_length * meta_batch_size)     # meta_batch_size = num of tasks
                                                                                    # (sampled from classes, "trials")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='Random seed to use for reproducibility')
    parser.add_argument('--epochs', type=int, default=300, help='Epochs to execute')
    parser.add_argument('--episodes_per_task', type=int, default=10, help='Number of episodes to sample per task')
    parser.add_argument('--meta_batch_size', type=int, default=10,
                        help='Tasks which are sampled per rollout (=trials in the original RL2 paper)')

    args = parser.parse_args()
    ml10_rl2_ppo(seed=args.seed, epochs=args.epochs, episodes_per_task=args.episodes_per_task,
                 meta_batch_size=args.meta_batch_size)
