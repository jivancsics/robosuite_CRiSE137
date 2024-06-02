from garage import wrap_experiment
from garage.experiment.deterministic import set_seed
from garage.trainer import TFTrainer


@wrap_experiment(snapshot_mode='gap', snapshot_gap=10, archive_launch_repo=False)
def pre_trained_crise3_rl2_ppo(ctxt,
                               snapshot_dir='IIWA14_extended_nolinear/data/local/experiment/singleml_rl2_ppo',
                               seed=1):
    set_seed(seed)
    with TFTrainer(snapshot_config=ctxt) as trainer:
        trainer.restore(snapshot_dir)
        trainer.resume()

if __name__ == "__main__":
    pre_trained_crise3_rl2_ppo()
