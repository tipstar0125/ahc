from __future__ import annotations

import os
import optuna
import subprocess
import threading
from multiprocessing import cpu_count
from pathlib import Path
import joblib

from common.settings import TEST_PATH

test_case_num = 48
path = Path(TEST_PATH)


def exec(file_num: int, params: dict[str, int]) -> int:
    env = os.environ.copy()
    for k, v in params.items():
        env[k] = str(v)
    filename = str(file_num).zfill(3)
    cmd = f"cargo run --bin a --features local < tools/in/in{filename}.txt > tools/out/{filename}.txt"
    proc = subprocess.Popen(cmd, shell=True, cwd=path, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stderr_list = proc.communicate()

    score = 0
    for stderr in stderr_list:
        out_list = stderr.decode().split("\n")

        for out in out_list:
            if "score" in out.lower():
                score = int(out.split()[1])
    return score


def worker(params: dict[str, int]) -> int:
    cores = cpu_count()
    scores = joblib.Parallel(n_jobs=cores)(joblib.delayed(exec)(i + 1, params) for i in range(test_case_num))
    score = int(sum(scores) / test_case_num)
    return score


def objective(trial: optuna.trial.Trial):
    env_name = "AHC_PARAMS_TEMP"
    temp = trial.suggest_int(env_name, 0, int(1e5))
    params = {env_name: temp}
    score = worker(params)
    return score


def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    print(study.best_params)
    print(study.best_value)


if __name__ == "__main__":
    main()
