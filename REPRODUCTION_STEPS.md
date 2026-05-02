# REPRODUCTION_STEPS — Team nlab

> Submissions are sent to the organizers as **fork URL + branch + note** (a description of how to run your model). This file is the recommended structure for your **note** — copy it to your repository root as `REPRODUCTION_STEPS.md`, fill every section, commit it, and link (or paste) it as your note. The evaluator will follow it **literally**, so commands and paths must be correct. Placeholders in parentheses are guidance — replace with your actual values.
>
> 日本語版: [REPRODUCTION_STEPS.template_ja.md](REPRODUCTION_STEPS.template_ja.md)

---

## 1. Overview

| Item | Value |
|---|---|
| Team name | nlab |
| Representative | takara taniguchi |
| Submission date | 2026-05-01(AoE) |
| Model summary | finetuned baseline |
| Framework | OpenPI |
| Repository | `https://github.com/taniguchitakara/airoa-evaluation-ICRA` |
| Branch | `feat/my-policy-openpi` |
| Commit hash | `abcd1234…` (optional but recommended) |
| Checkpoint S3 path | `s3://airoa-icra-team-15/my_checkpoint` |
| Expected VRAM | ~18 GB |

---

## 2. Prerequisites

- NVIDIA GPU with ≥ 16 GB VRAM (**Blackwell-compatible**: RTX 5070 Ti / compute 12.0)
- Docker Engine + Docker Compose v2
- NVIDIA Container Toolkit
- `rclone` (for checkpoint download from S3)
- External credentials required (e.g. `HF_TOKEN`, S3 credentials)? **State "none required" if not applicable.**

> If your model relies on anything not bundled in the repo/image at build time (gated HuggingFace models, external services, etc.), list every external resource the evaluator must access. The evaluation environment may be offline/locked.

---

## 3. Reproduction Steps

### 3.1 Clone and checkout

```bash
git clone https://github.com/taniguchitakara/airoa-evaluation-ICRA.git
cd airoa-evaluation-ICRA
git checkout feat/my-policy-openpi
git rev-parse HEAD  # optional: should match the commit hash in §1
```

### 3.2 Download checkpoint

The sample checkpoint is available at:

```bash
# Option 1: Using rclone (recommended)
mkdir -p src/my_policy
rclone copy s3://airoa-icra-team-15/my_checkpoint ./src/my_policy/my_checkpoint
```

### 3.3 Environment variables

The harness reads only three variables. List the ones you need:

```bash
export POLICY_CHECKPOINT_PATH=$(pwd)/src/my_policy/my_checkpoint
export POLICY_PYTORCH_DEVICE=cuda                             # optional
export POLICY_CONFIG_NAME="pi05_hsr_finetune"
```

If your server implementation reads any additional variables, list every one here (and make sure your `docker-compose.yml` / `.env` set them).

### 3.4 Start containers

```bash
./RUN-DOCKER-CONTAINER.sh up
```

### 3.5 Verify

```bash
# Wait for the policy server to be ready:
until curl -s http://localhost:8000/healthz 2>/dev/null | grep -q OK; do sleep 5; done
echo "READY"

# GPU usage sanity check:
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```

### 3.6 Stop

```bash
./RUN-DOCKER-CONTAINER.sh down
```

---

## 4. Files modified relative to the base repo

List paths you edited beyond the core scope (`server/`, `src/`) and explain why.

| Path | Reason |
|---|---|
| `server/serve_hsr_policy_ws.py` | (e.g. loads MyPolicyAdapter instead of the default OpenPI loader) |
| `server/Dockerfile` | (e.g. added torch 2.5.1 + transformers 4.46.0 + custom deps) |
| `src/<your_policy>/` | (e.g. new model implementation) |
| `docker-compose.yml` | (e.g. added one env var your server reads) |
| `client/Dockerfile` | (e.g. CUDA 12.8.1 base for Blackwell compatibility) |

---

## 5. Important notes

Anything the evaluator must know that is not obvious from the commands above. Examples:

- (e.g. "first startup takes ~2 min due to weight loading")
- (e.g. "tokenizer is bundled at `<path>`, no `HF_TOKEN` required")
- (e.g. "model compile is disabled to avoid 300s first-inference timeout")

---

## 6. Checkpoint file layout

**OpenPI-style:**

```
my_checkpoint/
├── assets
├── _CHECKPOINT_METADATA
├── params
└── train_state
```

Total size and additional details:

```
(paste your `tree` or `ls` output here)
```

Total size: ~X GB

---

## 7. Smoke test expected output

```
[INFO] [1777685767.977343]: Language instruction: Grasp the apple.
[INFO] [1777685767.979970]: Action: [-0.09246958 -0.29546636 -0.07545501  0.07828158 -0.38658898 -0.18507129
  0.13670256  0.16160193  0.00120045  0.00101012  0.09707234]
[INFO] [1777685768.069081]: Action executed.
[INFO] [1777685768.070608]: Language instruction: Grasp the apple.
[INFO] [1777685768.072403]: Action: [-8.36987617e-01  2.48983090e-01 -9.11006314e-02 -5.41954716e-02
  1.50849275e-01 -1.87482268e-01 -6.78847310e-01  9.79919888e-02
 -2.26931115e-05  6.77417847e-04  1.06662489e-01]
[INFO] [1777685768.169302]: Action executed.
[INFO] [1777685768.170858]: Language instruction: Grasp the apple.
[INFO] [1777685768.172623]: Action: [-5.56760268e-01  1.01885370e-01 -5.98885235e-01  4.83095874e-01
  6.32152848e-01 -1.91333979e-01  1.38033671e-01  4.21701370e-01
  1.24661939e-03  6.07066206e-04  1.12609714e-01]
[INFO] [1777685768.269304]: Action executed.
[INFO] [1777685768.270882]: Language instruction: Grasp the apple.
[INFO] [1777685768.272668]: Action: [-5.29337803e-01 -4.32714501e-02 -1.27289498e+00  1.50905967e-01
 -2.36704538e-01 -1.97564229e-01  4.02587720e-01 -1.45444886e-01
  8.81438493e-04  7.32054410e-04  1.10381812e-01]
[INFO] [1777685768.369301]: Action executed.
[INFO] [1777685768.370882]: Language instruction: Grasp the apple.
[INFO] [1777685768.372642]: Action: [ 1.00650157e+00  5.68600831e-03 -4.04673013e-01  5.35975456e-01
 -5.96296794e-01 -1.98895767e-01  5.96039654e-01 -1.34861296e-01
  3.05418973e-04  6.98853342e-04  1.07818596e-01]
[INFO] [1777685768.469301]: Action executed.
[INFO] [1777685768.470858]: Language instruction: Grasp the apple.
[INFO] [1777685768.472673]: Action: [-3.24975949e-01  8.67282930e-02  6.49093191e-01 -9.63692022e-01
  5.92806204e-01 -2.02976480e-01 -2.59924627e-01 -1.41252952e+00
  6.54695614e-04  8.45194154e-04  9.66115817e-02]
[INFO] [1777685768.721715]: Action executed.
[INFO] [1777685768.722723]: Language instruction: Grasp the apple.
[INFO] [1777685768.723481]: Action: [-2.39688339e-01 -1.29832878e-01  2.32059130e-02 -7.15727483e-01
 -4.99993531e-01  1.03935987e-01  1.64779445e-01 -1.62862997e-01
  1.07077509e-03  5.70506527e-05  1.46058267e-02]
```

---

## 8. Contact

- Team: (team name)
- Representative: (name) <email>
- Submission date: YYYY-MM-DD
