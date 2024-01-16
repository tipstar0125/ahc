from __future__ import annotations

import subprocess
import re
from pathlib import Path

from common.settings import TEST_PATH


def main():
    if TEST_PATH is None:
        return

    path = Path(TEST_PATH)

    with open("../cargo_multi_test/score.txt", encoding="UTF-8") as f:
        read_lines = f.readlines()
    score_list = []
    for line in read_lines:
        score_list.append(int(line))

    ok_cnt = 0
    test_case_num = 1000
    score_sum = 0
    for i in range(test_case_num):
        filename = str(i).zfill(4)
        cmd = f"cargo run --release --bin vis in/{filename}.txt out/{filename}.txt"
        proc = subprocess.Popen(cmd, shell=True, cwd=path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout_list = proc.communicate()
        score_bytes = stdout_list[0]
        score = int(re.sub(r"\D", "", str(score_bytes)))
        score_sum += score
        print(f"{filename} score: {score}")
        # print(score, score_list[i])
        if score == score_list[i]:
            ok_cnt += 1
    print(f"Score sum: {score_sum}, {ok_cnt}/{test_case_num}")


if __name__ == "__main__":
    main()
