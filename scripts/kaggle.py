"""Run repo commands on a free Kaggle GPU, so heavy training does not load the Mac.

    uv run python scripts/kaggle.py data                         # upload data/ as a private Kaggle dataset
    uv run python scripts/kaggle.py push NAME "cmd" ["cmd" ...]   # run commands at the current pushed commit
    uv run python scripts/kaggle.py status NAME
    uv run python scripts/kaggle.py pull NAME                    # download the log and copy results/*.json here

Needs the kaggle CLI (uv tool install kaggle) and a Kaggle API token in ~/.kaggle/access_token.
Each job clones the GitHub repo at the current commit, so that commit must be pushed first.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DATA_FILES = ["beauty.json", "text_emb.npy", "semantic_ids_loo.json", "semantic_ids_time.json"]
DATASET = "semantic-id-recs-data"
ACCELERATOR = "NvidiaTeslaT4"

# Each GPU takes the next command from a shared queue, so a T4 x2 machine runs two commands at once.
RUN_PY = """\
import glob, os, queue, shutil, subprocess, sys, threading
import torch
subprocess.run(["git", "clone", "-q", REPO, "repo"], check=True)
os.chdir("repo")
subprocess.run(["git", "checkout", "-q", COMMIT], check=True)
shutil.copytree(os.path.dirname(glob.glob("/kaggle/input/**/beauty.json", recursive=True)[0]), "data")
os.makedirs("results", exist_ok=True)
os.makedirs("/kaggle/working/logs", exist_ok=True)
subprocess.run("nvidia-smi -L; python -c 'import torch, transformers; print(torch.__version__, transformers.__version__)'", shell=True)

todo, failed = queue.Queue(), []
for cmd in COMMANDS:
    todo.put(cmd)

def worker(gpu):
    while not todo.empty():
        cmd = todo.get()
        name = cmd.replace("uv run ", "").replace("python ", "").replace(" --", "_").replace(" ", "_").replace(".py", "")
        print(f"[gpu {gpu}] start {cmd}", flush=True)
        with open(f"/kaggle/working/logs/{name}.log", "w") as log:
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
            for line in p.stdout:
                log.write(line)
                log.flush()
                print(f"[gpu {gpu}] {line}", end="", flush=True)
        if p.wait():
            failed.append(cmd)
        shutil.copytree("results", "/kaggle/working/results", dirs_exist_ok=True)

threads = [threading.Thread(target=worker, args=(g,)) for g in range(max(1, torch.cuda.device_count()))]
for t in threads:
    t.start()
for t in threads:
    t.join()
sys.exit(f"failed: {failed}" if failed else 0)
"""


def sh(*cmd, check=True):
    return subprocess.run(cmd, check=check, capture_output=True, text=True).stdout.strip()


def user():
    return next(line.split(": ")[1] for line in sh("kaggle", "config", "view").splitlines() if "username" in line)


def kernel_id(name):
    return f"{user()}/semantic-id-recs-{name}"


def upload_data():
    folder = Path(tempfile.mkdtemp())
    for f in DATA_FILES:
        shutil.copy(Path("data") / f, folder)
    dataset_id = f"{user()}/{DATASET}"
    (folder / "dataset-metadata.json").write_text(json.dumps({"title": DATASET, "id": dataset_id, "licenses": [{"name": "other"}]}))
    exists = subprocess.run(["kaggle", "datasets", "status", dataset_id], capture_output=True).returncode == 0
    cmd = ["kaggle", "datasets", "version", "-p", folder, "-m", "update"] if exists else ["kaggle", "datasets", "create", "-p", folder]
    subprocess.run(cmd, check=True)


def push(name, commands):
    sh("git", "fetch", "-q", "origin")
    if subprocess.run(["git", "merge-base", "--is-ancestor", "HEAD", "origin/main"]).returncode:
        sys.exit("push the current commit to GitHub first, the Kaggle job clones it from there")
    header = f"REPO = {sh('git', 'remote', 'get-url', 'origin')!r}\nCOMMIT = {sh('git', 'rev-parse', 'HEAD')!r}\nCOMMANDS = {commands!r}\n"
    folder = Path(tempfile.mkdtemp())
    (folder / "run.py").write_text(header + RUN_PY)
    slug = kernel_id(name).split("/")[1]
    meta = {
        "id": kernel_id(name), "title": slug, "code_file": "run.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [f"{user()}/{DATASET}"], "competition_sources": [], "kernel_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta))
    subprocess.run(["kaggle", "kernels", "push", "-p", folder, "--accelerator", ACCELERATOR], check=True)


def pull(name):
    out = Path("runs") / name
    subprocess.run(["kaggle", "kernels", "output", kernel_id(name), "-p", out, "-o"], check=True)
    for f in (out / "results").glob("*.json"):
        shutil.copy(f, Path("results") / f.name)
        print("copied", f.name)


if __name__ == "__main__":
    action, *rest = sys.argv[1:]
    if action == "data":
        upload_data()
    elif action == "push":
        push(rest[0], rest[1:])
    elif action == "status":
        subprocess.run(["kaggle", "kernels", "status", kernel_id(rest[0])])
    elif action == "pull":
        pull(rest[0])
    else:
        sys.exit(__doc__)
