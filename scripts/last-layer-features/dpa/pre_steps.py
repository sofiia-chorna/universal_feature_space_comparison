import os
import subprocess


def run(cmd, cwd=None):
    print(f"Running: {cmd} (cwd={cwd})")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.join(script_dir, "deepmd-kit")

    if not os.path.exists(repo_dir):
        print("Cloning repo: deepmd-kit on branch devel")
        run(
            "git clone https://github.com/deepmodeling/deepmd-kit.git -b devel",
            cwd=script_dir,
        )

    run("pip install -v -e . --verbose", cwd=repo_dir)

    PROJECT_ROOT = os.path.abspath(os.path.join(script_dir, "../../.."))

    print("Getting checkpoint: DPA-3.1-3M.pt")
    run(
        "wget https://store.aissquare.com/models/35b4ce45-4f59-4868-9fd7-a0c0f5ad9464/DPA-3.1-3M.pt",
        cwd=script_dir,
    )

    branches = {
        "Omat24": "Omat24",
        "Mptraj": "MP_traj_v024_alldata_mixu",
        "ODAC23": "ODAC23",
        "OC20M": "OC20M",
        "SPICE2": "SPICE2",
    }

    for model_name, branch_name in branches.items():
        run(
            f"dp --pt freeze -c DPA-3.1-3M.pt -o dpa3.1_{model_name}.pth --model-branch {branch_name}",
            cwd=script_dir,
        )

        MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
        os.makedirs(MODELS_DIR, exist_ok=True)

        run(
            f"mv DPA-3.1-3M.pt {os.path.join(MODELS_DIR, 'DPA-3.1-3M.pt')}",
            cwd=script_dir,
        )
        run(
            f"mv dpa3.1_{model_name}.pth {os.path.join(MODELS_DIR, 'dpa3.1_{model_name}.pth')}",
            cwd=script_dir,
        )
