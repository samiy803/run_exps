import argparse, glob, json, os, pathlib, zipfile
from functools import partial

import gymnasium as gym
import highway_env

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback


import wandb
from wandb.integration.sb3 import WandbCallback

ALGOS = {"PPO": PPO, "TD3": TD3, "SAC": SAC}


class OptunaPruningCallback(BaseCallback):
    """Reports episodic return to Optuna and prunes under‑performing trials."""

    def __init__(self, trial, eval_env, eval_freq, n_eval_episodes=1):
        super().__init__(verbose=0)
        self.trial = trial
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self._next_eval = eval_freq

    def _on_step(self):
        if self.num_timesteps >= self._next_eval:
            m, _ = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes, warn=False)
            self.trial.report(m, step=self.num_timesteps)
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            self._next_eval += self.eval_freq
        return True


def make_env(env_id, env_config, seed, rank):
    def _init():
        import highway_env 
        env = gym.make(env_id, config=env_config, render_mode=None)
        env.reset(seed=seed + rank)
        return env
    return _init


def objective(trial, cfg, tuning_ts, n_envs):
    """Objective function for Optuna hyperparameter tuning."""
    def suggest(space):
        """Convert search space to Optuna suggestions."""
        out = {}
        for k, spec in space.items():
            m = spec["method"]
            if m == "log_uniform":
                out[k] = trial.suggest_float(k, spec["low"], spec["high"], log=True)
            elif m == "uniform":
                out[k] = trial.suggest_float(k, spec["low"], spec["high"])
            elif m == "int_uniform":
                out[k] = trial.suggest_int(k, spec["low"], spec["high"])
            elif m == "categorical":
                out[k] = trial.suggest_categorical(k, spec["choices"])
            else:
                raise ValueError(f"Unknown suggest method: {m}")
        return out

    params = suggest(cfg["search_space"])

    env_fns = [make_env(cfg["env_id"], cfg.get("env_config", {}), seed=0, rank=i) for i in range(n_envs)]
    train_env = VecNormalize(SubprocVecEnv(env_fns), norm_obs=True, norm_reward=True, gamma=params.get("gamma", 0.99))
    eval_env  = VecNormalize(SubprocVecEnv(env_fns), norm_obs=True, norm_reward=False, gamma=params.get("gamma", 0.99))

    model = ALGOS[cfg["algorithm"]](
        "MlpPolicy",
        train_env,
        tensorboard_log=None, # No tensorboard logging for tuning
        verbose=0,
        **params,
    )

    eval_freq = max(1000, tuning_ts // 10)
    pruning_cb = OptunaPruningCallback(trial, eval_env, eval_freq)

    model.learn(tuning_ts, callback=pruning_cb, progress_bar=False)

    mean_r, _ = evaluate_policy(model, eval_env, n_eval_episodes=3, warn=False)
    train_env.close(); eval_env.close()
    return float(mean_r)


def run_single(cfg_path: str, upload: str, wandb_project: str):
    cfg = json.load(open(cfg_path))
    exp_name = pathlib.Path(cfg_path).stem
    out_dir = pathlib.Path("output") / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    n_envs = int(cfg.get("n_envs", 4))
    total_ts = int(cfg["total_timesteps"])
    tuning_ts = int(total_ts * cfg.get("tuning_fraction", 0.2))

    
    hb = cfg.get("hyperband", {})
    pruner = HyperbandPruner(
        min_resource=int(hb.get("min_resource", tuning_ts // 4)),
        max_resource=int(hb.get("max_resource", tuning_ts)),
        reduction_factor=int(hb.get("reduction_factor", 3)),
    )


    storage = f"sqlite:///{out_dir}/study.db"
    study = optuna.create_study(
        study_name=exp_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(),
        pruner=pruner,
    )

    study.optimize(partial(objective, cfg=cfg, tuning_ts=tuning_ts, n_envs=n_envs),
                   n_trials=cfg["n_trials"], n_jobs=1, show_progress_bar=False)

    
    (out_dir / "best_params.json").write_text(json.dumps(study.best_params, indent=2))
    study.trials_dataframe().to_csv(out_dir / "trials.csv", index=False)
    with open(out_dir / "study_info.json", "w") as f:
        json.dump({
            "study_name": study.study_name,
            "n_trials": len(study.trials),
            "best_trial": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": study.best_params,
        }, f, indent=2)

    # Train final model with best hyperparameters
    env_fns = [make_env(cfg["env_id"], cfg.get("env_config", {}), seed=42, rank=i) for i in range(n_envs)]
    final_env = VecNormalize(SubprocVecEnv(env_fns), norm_obs=True, norm_reward=True, gamma=study.best_params.get("gamma", 0.99))

    tb_dir = out_dir / "tensorboard"
    callbacks = []
    if upload == "wandb" and wandb is not None and WandbCallback is not None and wandb_project:
        run = wandb.init(project=wandb_project, name=exp_name, config=study.best_params, reinit=True)
        callbacks.append(WandbCallback(model_save_path=str(out_dir / "wandb_models"), verbose=0))
    else:
        run = None

    final_model = ALGOS[cfg["algorithm"]](
        "MlpPolicy",
        final_env,
        tensorboard_log=str(tb_dir),
        verbose=1,
        **study.best_params,
    )

    final_model.learn(total_timesteps=total_ts, callback=callbacks or None)
    final_model.save(out_dir / "final_model.zip")
    final_env.close()


    zip_path = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out_dir.rglob("*"):
            z.write(p, p.relative_to(out_dir))


    if upload == "wandb" and run is not None:
        art = wandb.Artifact(exp_name, type="rl-experiment")
        art.add_file(str(zip_path))
        run.log_artifact(art)
        run.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs_dir", required=True)
    ap.add_argument("--upload", choices=["none", "wandb"], default="none")
    ap.add_argument("--wandb_project")
    args = ap.parse_args()

    for cfg in sorted(glob.glob(os.path.join(args.configs_dir, "*.json"))):
        run_single(cfg, upload=args.upload, wandb_project=args.wandb_project)

if __name__ == "__main__":
    main()