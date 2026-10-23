# Motion Streamer setup

Run the main repository's `setup.sh` in `unitree_rl_mjlab` first. The streamer keeps its ARDY environment separate from training. The expected workspace layout is:

```text
workspace/
  softmimic_mjlab/       # this repository, including both demos
  ardy/                 # upstream source, pinned below
  sonic23/              # existing cache/environment directory; no SONIC training is required
    .venv-ardy/
    checkpoints/
    ardy_text_models.json
    ardy_motions/
```

The `sonic23` name is retained for the existing model/library cache. No SONIC policy checkpoint is required for text generation or retargeting.

From the repository root, with an existing Hugging Face account authorized to download `meta-llama/Meta-Llama-3-8B-Instruct`:

```bash
conda activate unitree_rl_mjlab
```

```bash
bash demos/ardy_web/setup.sh
```

```bash
python demos/sonic_web/prepare.py --reference datasets/streamer/stand.csv \
  --deploy demos/policies/stand_20000/params/deploy.yaml \
  --policy demos/policies/stand_20000/exported/policy.onnx
```

```bash
bash demos/ardy_web/run.sh
```

Setup clones ARDY at `e706e27acc88070918308a10922b8f7174d0825e`, creates a separate venv using the installed Torch runtime, installs ARDY and Mink, and downloads pinned motion/text dependencies. It never accepts model license terms; access errors must be resolved in the user's Hugging Face account. An existing ARDY checkout at another revision is preserved and causes setup to stop.

The motion model is `nvidia/ARDY-G1-RP-25FPS-Horizon8` at `fb815e8021d700af523ac32d9851b7e489fbb60b`. The downloader pins the Llama and LLM2Vec adapter revisions and records resolved paths and sizes in `../sonic23/ardy_text_models.json`. Text weights occupy about 15.29 GiB; motion files about 0.72 GiB, excluding the environment. Dependency versions from the existing tested installation are recorded in `requirements-tested.txt`; different base Torch/CUDA installations need revalidation.

On this workstation all dependencies are already downloaded. Start only one server per port:

```bash
bash demos/ardy_web/run.sh
```

It binds to `127.0.0.1:8089`, disables CUDA, uses low CPU thread counts and offline model loading. First-time prompt encoding can take substantially longer than subsequent cached prompts. Generation is not real-time on this CPU. Saved clips and audits live outside the Git checkout in `../sonic23/ardy_motions/`.
