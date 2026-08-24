# Machine Learning Engineer Fundamentals
## Class 2: MLE Tooling Practices

---

## Table of Contents
1. [Recap & Class 2 Overview](#1-recap--class-2-overview)
2. [Code Version Control (Git) for ML](#2-code-version-control-git-for-ml)
3. [Data Version Control (DVC)](#3-data-version-control-dvc)
4. [Code Review Best Practices](#4-code-review-best-practices)
5. [Model Cards & Documentation](#5-model-cards--documentation)
6. [Course Project: Standardize Your ML Project](#6-course-project-standardize-your-ml-project)
7. [Tooling Cheat Sheet](#7-tooling-cheat-sheet)
8. [Course Discussion](#8-course-discussion)
9. [Homework / Next Steps](#9-homework--next-steps)

---

## 1. Recap & Class 2 Overview

### Where We Left Off

Class 1 covered the MLE role in industry, the required skillset, how MLEs compare
to DS/DE/AI Engineer roles, agile process, the end-to-end training/deployment
pipeline, interviewing, and the latest CV/multimodal trends. This class turns
from "what MLEs do" to "how MLEs actually work day to day" — the tooling that
turns a notebook full of experiments into something a team can trust, review,
and ship.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     WHY TOOLING IS ITS OWN CLASS                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  A model that only exists on one laptop, in one notebook, with no          │
│  versioned data and no review trail, is a liability — not an asset.        │
│  Good tooling is what makes ML work REPRODUCIBLE, REVIEWABLE, and          │
│  TRANSFERABLE across a team.                                               │
│                                                                             │
│   Without Tooling                    │  With Tooling                       │
│   ────────────────                   │  ─────────────                      │
│   • "It works on my machine"         │  • Pinned env + versioned data      │
│   • Untracked data.csv v2_final_final│  • DVC-tracked dataset revisions    │
│   • One giant notebook               │  • Modular, reviewed, tested code   │
│   • No record of why a model shipped │  • Model card + experiment log      │
│   • Silent regressions               │  • CI catches them before merge     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Class 2 Roadmap

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CLASS 2 SESSION FLOW (60 min)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  0:00 ─ 0:05   Recap of Class 1 + today's goals                            │
│  0:05 ─ 0:20   Code version control (Git) for ML projects                  │
│  0:20 ─ 0:35   Data version control (DVC)                                  │
│  0:35 ─ 0:45   Code review best practices                                  │
│  0:45 ─ 0:55   Model cards & documentation                                 │
│  0:55 ─ 1:00   Project kickoff: standardize your project + Q&A            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Learning Objectives

By the end of this class, you should be able to:

| Objective | You Will Be Able To |
|-----------|---------------------|
| **Git for ML** | Set up a branching strategy and commit hygiene suited to experimental ML work |
| **DVC** | Version large datasets/models alongside Git and build a reproducible DVC pipeline |
| **Code Review** | Run/participate in an ML-aware code review using a concrete checklist |
| **Documentation** | Write a model card and a project README that a new teammate could onboard from |
| **Project** | Apply all of the above to standardize one of your own academic/personal projects |

---

## 2. Code Version Control (Git) for ML

### Why Git Alone Isn't Enough — But Is Still Non-Negotiable

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  WHAT GIT VERSIONS WELL vs POORLY (FOR ML)                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  GIT IS GREAT FOR                    │  GIT IS BAD FOR                     │
│  ─────────────────                   │  ───────────────                    │
│  • Source code (.py, .cpp)           │  • Large datasets (images, video)   │
│  • Configs (.yaml, .json)            │  • Model weight files (.pt, .onnx)  │
│  • Small fixtures / test data        │  • Binary blobs that change often   │
│  • Documentation                     │  • Unstructured notebook diffs      │
│  • Infra-as-code                     │                                     │
│                                                                             │
│  → Git handles code. DVC (Section 3) handles data & model artifacts.       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### ML-Adapted Branching Strategy

Most ML teams adapt trunk-based/GitFlow-style branching to account for
long-running experiments that may never ship.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ML GIT BRANCHING MODEL                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   main ────●─────────────●─────────────────●───────────────────▶          │
│            │             │                 │                              │
│            │             │                 │  (tagged releases: v1.0,     │
│            │             │                 │   v1.1 — deployed models)    │
│            │             │                 │                              │
│   develop ─┴──●───●──────┴────●───●────────┴──●─────────────────▶         │
│               │   │           │   │           │                           │
│               │   │           │   │           │                           │
│   feature/    └───┘           │   │           │                           │
│   data-loader                 │   │           │                           │
│                                │   │           │                           │
│   exp/depth-vit-backbone ──────┘   │           │  (short-lived, may be    │
│                                     │           │   deleted if it fails)  │
│   exp/loss-fn-scale-invariant ──────┘           │                        │
│                                                  │                         │
│   fix/dataloader-memory-leak ────────────────────┘                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

BRANCH NAMING CONVENTIONS
──────────────────────────────────────────────────────────────────────────────
feature/<short-desc>     New capability that will ship (e.g. feature/dvc-setup)
exp/<hypothesis>         Exploratory experiment, may be abandoned
fix/<bug>                Bug fix
release/<version>        Release stabilization branch
```

### Why Experiment Branches Matter

Unlike feature branches, `exp/*` branches are expected to fail — that is
normal, not a problem. Merging an experiment branch means "this hypothesis
was validated and should become part of the pipeline," not just "this code
runs."

```
EXPERIMENT BRANCH LIFECYCLE
═══════════════════════════════════════════════════════════════

  exp/vit-backbone
       │
       ▼
  Run training + log to MLflow/W&B ──▶ Compare against baseline metric
       │                                        │
       ├── Improves metric ──▶ Open PR into develop, link experiment run
       │                                        │
       └── No improvement  ──▶ Document learning in experiment log, delete branch
```

### Commit Message Conventions

```
CONVENTIONAL COMMITS FOR ML PROJECTS
═══════════════════════════════════════════════════════════════

<type>(<scope>): <short summary>

<optional body: what changed and WHY>
<optional footer: linked experiment ID, issue #, breaking change note>

┌──────────┬────────────────────────────────────────────────────┐
│ Type     │ Use For                                             │
├──────────┼────────────────────────────────────────────────────┤
│ feat     │ New model, new pipeline stage, new feature          │
│ fix      │ Bug fix (data leak, wrong metric, off-by-one)       │
│ data     │ Dataset version bump / data pipeline change         │
│ exp      │ Experiment-only change, not meant to persist        │
│ perf     │ Latency/throughput/memory optimization              │
│ refactor │ No behavior change, code structure only             │
│ docs     │ README, model card, docstrings                      │
│ test     │ Adding/fixing tests                                 │
│ chore    │ Tooling, CI, dependency bumps                       │
└──────────┴────────────────────────────────────────────────────┘

EXAMPLE:
─────────
feat(depth-model): add scale-invariant loss for monocular depth

Replaces plain MSE with scale-invariant log loss (Eigen et al.)
to reduce sensitivity to global scale ambiguity in monocular setups.

AbsRel improved 0.18 → 0.14 on NYU-Depth-V2 val split.
Experiment: exp_2026_014 (see MLflow run 7a2f9c)
```

### .gitignore Starter for ML Projects

```
# .gitignore — ML project starter
──────────────────────────────────────────────────────────────────
# Environments
.venv/
__pycache__/
*.pyc

# Data & model artifacts (tracked by DVC instead — see Section 3)
data/
!data/.gitkeep
*.pt
*.onnx
*.ckpt
mlruns/

# Notebooks — keep source, strip output (see nbstripout below)
.ipynb_checkpoints/

# Secrets & local config
.env
*.key
config/local.yaml

# OS / editor
.DS_Store
.vscode/
```

### Handling Notebooks in Git

Notebooks are one of the most common sources of unreviewable diffs and merge
conflicts on ML teams, because `.ipynb` files store cell outputs and
execution counts as JSON alongside the code.

```
NOTEBOOK HYGIENE FOR GIT
═══════════════════════════════════════════════════════════════

PROBLEM: A notebook diff looks like this after just re-running a cell:
  - "execution_count": 12,
  + "execution_count": 13,
  - "outputs": [{"data": {"image/png": "iVBORw0KGgoAAAANSU..." (huge)

SOLUTIONS (pick one, ranked by team maturity):
┌────────────────────┬──────────────────────────────────────────┐
│ nbstripout         │ Git filter that strips outputs on commit  │
│ (pre-commit hook)  │ automatically — cheapest to adopt          │
├────────────────────┼──────────────────────────────────────────┤
│ jupytext           │ Pairs .ipynb with a plain .py/.md mirror   │
│                    │ that git actually diffs cleanly            │
├────────────────────┼──────────────────────────────────────────┤
│ "Notebooks are for │ Treat notebooks as scratch/EDA only;       │
│  exploration only" │ production code lives in reviewed .py     │
│                    │ modules that notebooks import               │
└────────────────────┴──────────────────────────────────────────┘

$ pip install nbstripout
$ nbstripout --install       # registers a git filter in this repo
```

### Git Command Cheat Sheet for ML Workflows

```
DAILY GIT WORKFLOW
──────────────────────────────────────────────────────────────────────────────
$ git checkout develop && git pull
$ git checkout -b exp/depth-vit-backbone
$ git add src/models/vit_depth.py
$ git commit -m "exp(depth-model): try ViT-B backbone vs ResNet50"
$ git push -u origin exp/depth-vit-backbone

INSPECTING HISTORY
──────────────────────────────────────────────────────────────────────────────
$ git log --oneline --graph --all         # visualize branch history
$ git blame src/train.py                  # who/when changed a line, and why
$ git bisect start                        # binary-search for a regression

CLEANING UP EXPERIMENT BRANCHES
──────────────────────────────────────────────────────────────────────────────
$ git branch --merged develop | grep 'exp/' | xargs git branch -d
$ git push origin --delete exp/depth-vit-backbone

RESOLVING NOTEBOOK/CONFIG MERGE CONFLICTS
──────────────────────────────────────────────────────────────────────────────
$ git checkout --ours  config/params.yaml   # keep my version
$ git checkout --theirs config/params.yaml  # keep incoming version
```

---

## 3. Data Version Control (DVC)

### Recap: Why Data Needs Its Own Version Control

As introduced in Class 1, datasets and model weights are too large for Git
and change independently of code. DVC solves this by storing lightweight
pointer files in Git while the actual bytes live in remote storage
(S3/GCS/Azure/local NFS).

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DVC + GIT WORKFLOW (RECAP)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   LOCAL WORKSPACE                        REMOTE STORAGE                 │
│   ───────────────                        ──────────────                 │
│                                                                         │
│   ┌─────────────┐    dvc add            ┌─────────────┐                │
│   │  data/      │ ──────────────────▶   │   S3/GCS/   │                │
│   │  images/    │    dvc push           │   Azure     │                │
│   │  (large)    │ ◀──────────────────   │   Blob      │                │
│   └─────────────┘    dvc pull           └─────────────┘                │
│         │                                                               │
│         ▼                                                               │
│   ┌─────────────┐    git add/commit     ┌─────────────┐                │
│   │ data.dvc    │ ──────────────────▶   │   GitHub/   │                │
│   │ (pointer)   │                       │   GitLab    │                │
│   │ (~1KB)      │                       │             │                │
│   └─────────────┘                       └─────────────┘                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Setting Up DVC in a New Project

```
INITIAL SETUP
──────────────────────────────────────────────────────────────────────────
$ git init                                     # if not already a git repo
$ dvc init                                     # creates .dvc/ directory
$ git add .dvc .dvcignore && git commit -m "chore: initialize DVC"

$ dvc remote add -d storage s3://my-bucket/dvc-storage
$ git add .dvc/config && git commit -m "chore: configure DVC remote"

TRACKING A DATASET
──────────────────────────────────────────────────────────────────────────
$ dvc add data/training_images/
  → creates data/training_images.dvc   (pointer, checked into git)
  → adds data/training_images/ to .gitignore automatically

$ git add data/training_images.dvc .gitignore
$ git commit -m "data: add v1 of training image set (12,400 images)"
$ dvc push                                     # uploads bytes to remote

ON ANOTHER MACHINE / TEAMMATE'S CLONE
──────────────────────────────────────────────────────────────────────────
$ git clone <repo> && cd <repo>
$ dvc pull                                     # downloads data referenced
                                                # by the .dvc pointer files
```

### DVC Pipelines: Making Preprocessing → Train → Eval Reproducible

A `dvc.yaml` pipeline declares stages, their dependencies, parameters, and
outputs — so `dvc repro` can re-run only what's stale, like a Makefile for ML.

```
dvc.yaml — MULTI-STAGE PIPELINE EXAMPLE
──────────────────────────────────────────────────────────────────────────
stages:
  prepare:
    cmd: python prepare_data.py
    deps:
      - raw_data/
      - prepare_data.py
    outs:
      - processed_data/

  train:
    cmd: python train.py --config params.yaml
    deps:
      - processed_data/
      - train.py
    params:
      - learning_rate
      - batch_size
    outs:
      - models/depth_model.pt
    metrics:
      - metrics.json:
          cache: false

  evaluate:
    cmd: python evaluate.py
    deps:
      - models/depth_model.pt
      - test_data/
    metrics:
      - eval_metrics.json:
          cache: false

RUNNING THE PIPELINE
──────────────────────────────────────────────────────────────────────────
$ dvc repro                    # runs only stages whose deps/params changed
$ dvc dag                      # visualize the pipeline DAG
$ dvc metrics show             # compare current metrics
$ dvc metrics diff main        # compare metrics against another branch/commit
```

### DVC Experiment Tracking (`dvc exp`)

```
LIGHTWEIGHT EXPERIMENT TRACKING WITHOUT A SEPARATE SERVER
──────────────────────────────────────────────────────────────────────────
$ dvc exp run --set-param train.lr=0.001
$ dvc exp run --set-param train.lr=0.0005
$ dvc exp show                 # table comparing all runs' params + metrics
$ dvc exp apply exp-a1b2c      # promote the best experiment to your workspace
$ dvc exp branch exp-a1b2c exp/best-lr   # turn it into a real git branch
```

### Data Versioning Tool Comparison

| Tool | Best For | Storage Model | Learning Curve |
|------|----------|---------------|-----------------|
| **DVC** | Small-mid teams, Git-native workflow | Pointer files in Git + remote blob store | Low |
| **Git LFS** | Occasional large files, simple history | Pointers in Git, LFS server for blobs | Low |
| **LakeFS** | Data-lake scale, S3-native versioning | Git-like branching directly on S3 | Medium |
| **Pachyderm** | Data pipelines as first-class citizen | Kubernetes-native, versioned data + pipelines | High |
| **Delta Lake** | Structured/tabular data at scale | Versioned Parquet + transaction log | Medium |

```
DECISION GUIDE
──────────────────────────────────────────────────────────────────────────
Small team, images/video, already using Git  ──▶ DVC
Only occasional large binary files            ──▶ Git LFS
Petabyte-scale data lake, S3-native           ──▶ LakeFS
Full pipeline orchestration + data versioning ──▶ Pachyderm
Structured/tabular analytics workloads        ──▶ Delta Lake
```

---

## 4. Code Review Best Practices

### Why ML Code Review Is Different

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              TRADITIONAL SWE REVIEW vs ML CODE REVIEW                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TRADITIONAL SOFTWARE                │  MACHINE LEARNING                    │
│  ─────────────────────               │  ─────────────────                   │
│  Bugs usually crash or throw         │  Bugs often produce a number that    │
│  visibly                             │  LOOKS plausible but is wrong        │
│                                                                             │
│  Correctness ≈ passes tests          │  Correctness ≈ passes tests AND      │
│                                       │  matches expected metric behavior    │
│                                                                             │
│  Deterministic given same input      │  Stochastic (seeds, GPU nondeterminism)│
│                                                                             │
│  Review focuses on logic & style     │  Review must also catch DATA LEAKAGE,│
│                                       │  metric misuse, and reproducibility  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### ML Code Review Checklist

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ML PULL REQUEST REVIEW CHECKLIST                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DATA CORRECTNESS                                                          │
│  ☐ Train/val/test splits don't overlap (no leakage)                        │
│  ☐ Normalization stats (mean/std) computed on train set only               │
│  ☐ No target leakage (features that encode the label)                      │
│  ☐ Data augmentation isn't applied to val/test sets                        │
│                                                                             │
│  REPRODUCIBILITY                                                           │
│  ☐ Random seeds are set (numpy, torch, python random)                      │
│  ☐ Config/hyperparameters are in a file, not hardcoded                     │
│  ☐ Dataset version (DVC hash) referenced in the experiment log             │
│  ☐ Environment is pinned (requirements.txt / lockfile / Docker image)      │
│                                                                             │
│  CORRECTNESS OF METRICS                                                    │
│  ☐ Metric matches the task (e.g. no accuracy on imbalanced classes)        │
│  ☐ Metric computed on the right split                                      │
│  ☐ Baseline comparison included, not just an absolute number               │
│                                                                             │
│  CODE QUALITY                                                              │
│  ☐ No dead/commented-out experiment code left in                           │
│  ☐ Functions are testable (not one 300-line script)                        │
│  ☐ Type hints on public functions                                          │
│  ☐ No hardcoded local file paths                                           │
│                                                                             │
│  DOCUMENTATION                                                             │
│  ☐ PR description explains the hypothesis and result, not just the diff    │
│  ☐ Non-obvious choices have a one-line comment explaining WHY              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### PR Template Example

```markdown
## What & Why
<!-- What experiment/change does this PR make, and what hypothesis is it testing? -->

## Results
| Metric   | Baseline | This PR | Δ      |
|----------|----------|---------|--------|
| AbsRel   | 0.18     | 0.14    | -0.04  |
| RMSE     | 0.62     | 0.55    | -0.07  |
| Latency  | 45ms     | 48ms    | +3ms   |

## Reproducibility
- Dataset version (DVC): `data/training_images.dvc @ a1b2c3d`
- Experiment tracking run: `mlflow run 7a2f9c` / `wandb run xk3f9d`
- Config: `configs/depth_vit.yaml`

## Checklist
- [ ] Seeds set / run is reproducible
- [ ] No data leakage between splits
- [ ] Tests added/updated
- [ ] Docs/model card updated if behavior changed
```

### The ML Review Workflow

```
ML CODE REVIEW FLOW
═══════════════════════════════════════════════════════════════

  Open PR ──▶ CI runs (lint, unit tests, data validation tests)
                  │
                  ▼
          Automated checks pass? ──No──▶ Fix and re-push
                  │ Yes
                  ▼
          Human review (checklist above)
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
  Changes requested    Approved
        │                   │
        ▼                   ▼
   Address comments    Merge to develop
                             │
                             ▼
                  Update experiment log / model registry
```

### Reviewing via Git: Branches, Diffs, and Merge Strategies

Reviewing "via git" means more than reading a diff in a browser tab.
Knowing the underlying commands matters more in ML review than in typical
software review, because you often need to actually *run* a teammate's
branch to trust a metric claim, not just read the code that produced it.

```
CHECKING OUT SOMEONE ELSE'S PR LOCALLY (GitHub)
──────────────────────────────────────────────────────────────────────────
$ git fetch origin pull/123/head:review-pr-123
$ git checkout review-pr-123
$ python -m src.train --epochs 1        # actually run it, don't just read it

COMPARING A BRANCH AGAINST ITS BASE
──────────────────────────────────────────────────────────────────────────
$ git diff main...exp/depth-vit-backbone            # only what the branch changed
$ git log --oneline main..exp/depth-vit-backbone    # commits the branch adds
$ git diff main...exp/depth-vit-backbone -- src/model.py   # scope to one file

INLINE SUGGESTIONS
──────────────────────────────────────────────────────────────────────────
GitHub's "suggest a change" on a diff line produces a commit the author can
accept with one click -- faster than prose comments for small, unambiguous
fixes (a renamed variable, a typo, an off-by-one).
```

**Branch protection** turns the checklist from Section 4 into something CI
actually enforces, instead of something reviewers have to remember:

```
BRANCH PROTECTION FOR develop / main  (GitHub: Settings → Branches)
──────────────────────────────────────────────────────────────────────────
☐ Require a pull request before merging (no direct pushes)
☐ Require at least 1 approving review
☐ Require status checks to pass before merging (lint, tests, smoke-train)
☐ Require branches to be up to date with the base before merging
☐ Do not allow bypassing the above, even for admins
```

**Merge strategy** matters for ML repos specifically because experiment
branches accumulate a lot of noisy, non-final commits:

| Strategy | What It Does | Best For |
|---|---|---|
| Merge commit | Keeps every commit + adds a merge commit | Long-lived branches with meaningful individual commits |
| Squash and merge | Collapses the branch into one commit on the base | `exp/*` branches -- "try lr=0.01", "try lr=0.001", "fix typo" become one clean commit |
| Rebase and merge | Replays commits onto the base, linear history | Small branches you haven't shared with anyone else |

For `exp/*` branches specifically, **squash and merge** is usually right:
nobody needs the archaeology of every failed hyperparameter guess in
`develop`'s permanent history -- they need one commit in the Conventional
Commits format from Section 2, with the experiment ID and final result.

### Common ML Anti-Patterns to Flag in Review

| Anti-Pattern | Why It's a Problem | Fix |
|---|---|---|
| Fitting a `StandardScaler` on the full dataset before splitting | Leaks test statistics into training | Fit only on train, transform val/test |
| One 500-line `train.py` doing everything | Impossible to unit test or reuse | Split into data/model/train/eval modules |
| `random_state` left unset | Results aren't reproducible between runs | Set and log a fixed seed everywhere |
| Copy-pasted preprocessing between train and serving code | Train/serve skew | Share one preprocessing module/package |
| Committing a trained model binary directly to Git | Bloats repo, no versioning story | Track via DVC or a model registry |

### Automating What You Can: Pre-commit Hooks

```
pre-commit CONFIG (.pre-commit-config.yaml)
──────────────────────────────────────────────────────────────────────────
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks: [{id: ruff}]
  - repo: https://github.com/psf/black
    hooks: [{id: black}]
  - repo: https://github.com/kynan/nbstripout
    hooks: [{id: nbstripout}]
```

### CI/CD with GitHub Actions for ML

A workflow file at `.github/workflows/ci.yml` runs on events like
`pull_request` or `push`. For ML repos, add two ML-specific jobs on top of
standard lint/test: pulling DVC-tracked data, and a **smoke-train** (one
epoch, a tiny batch) that catches a broken training loop in ~30 seconds
instead of after a full multi-hour run.

```yaml
# .github/workflows/ci.yml
name: ML CI

on:
  pull_request:
    branches: [develop, main]
  push:
    branches: [develop, main]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
      - run: pip install -r requirements.txt
      - run: ruff check .
      - run: pytest tests/ -v

  data-and-smoke-train:
    needs: lint-and-test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - name: Pull DVC-tracked data
        run: dvc pull
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
      - name: Smoke-train (1 epoch, catches a broken training loop)
        run: python -m src.train --epochs 1 --batch_size 4
```

These are exactly the checks the branch protection rule above enforces:
`develop`/`main` can't accept a merge until both jobs are green.

**Continuous deployment** on merge to `main` -- e.g. build and push a
serving image -- is the same idea, gated to one branch:

```yaml
# .github/workflows/cd.yml
name: Build and Push Model Image

on:
  push:
    branches: [main]

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}:latest
```

**Bonus -- metrics on the PR itself:** [iterative/cml](https://cml.dev)
(from the DVC team) turns a CI run's metrics into an automatic PR comment,
so a reviewer sees the model's actual numbers without leaving GitHub:

```yaml
      - uses: iterative/setup-cml@v2
      - name: Post metrics comment
        env:
          REPO_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          dvc metrics diff main --show-md > report.md
          cml comment create report.md
```

Try this against the `lab/` project from this course: its `dvc.yaml`
already writes `outputs/eval_metrics.json` as a `cache: false` metric,
which is exactly what `dvc metrics diff` reads.

---

## 5. Model Cards & Documentation

### What Is a Model Card and Why It Matters

Model cards (Mitchell et al., 2019, "Model Cards for Model Reporting")
standardize how a model's intended use, performance, and limitations are
communicated — to teammates, downstream consumers, and auditors.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHY MODEL CARDS EXIST                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  WITHOUT A MODEL CARD                │  WITH A MODEL CARD                   │
│  ─────────────────────               │  ──────────────────                  │
│  "Which dataset was this trained on?"│  Documented under Training Data      │
│  "Does this work on dark-skin tones?"│  Documented under Bias/Fairness      │
│  "What's the expected latency?"      │  Documented under Metrics            │
│  "Can I use this for medical imaging?"│  Documented under Intended/Out-of-Scope Use│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Model Card Template

```markdown
# Model Card: Monocular Depth Estimation v1.2

## Model Details
- Architecture: DPT-Hybrid (ViT-B backbone + convolutional decoder)
- Framework: PyTorch 2.x, exported to ONNX for deployment
- Trained by: [Team/Author], [Date]
- License: [License]

## Intended Use
- Primary use: relative depth estimation from a single RGB frame on mobile devices
- Out-of-scope use: metric (absolute-scale) depth for safety-critical navigation;
  not validated for autonomous vehicle obstacle distance

## Training Data
- NYU Depth V2 (indoor scenes, 120K frames), KITTI (outdoor, 93K frames)
- Data version (DVC): `data/nyu_kitti_merged.dvc @ 9f31ab2`
- Known gaps: underrepresented in low-light and adverse-weather scenes

## Evaluation Data & Metrics
| Metric | Value | Split |
|---|---|---|
| AbsRel | 0.14 | NYU-Depth-V2 val |
| RMSE | 0.55 | NYU-Depth-V2 val |
| δ < 1.25 | 0.87 | NYU-Depth-V2 val |
| Inference latency | 48ms | iPhone 13, ONNX Runtime, FP16 |

## Limitations & Bias
- Performance degrades on reflective/transparent surfaces (glass, mirrors)
- Trained predominantly on North American indoor/outdoor scenes
- Not evaluated for fairness across scene demographics (n/a for depth-only task,
  but flagged if used with person-detection downstream)

## Ethical Considerations
- If used for people-counting/tracking downstream, follow applicable privacy
  and consent requirements for the deployment context

## Caveats & Recommendations
- Recommend re-validating on target deployment domain before shipping
- Monitor for depth drift if camera hardware or FOV changes
```

### README Structure for ML Repos

```
PROJECT README SKELETON
──────────────────────────────────────────────────────────────────────────
# Project Name

## Overview            — one paragraph: what this does and why
## Architecture         — diagram + key components
## Project Layout       — directory tree with one-line descriptions
## Setup                — environment, dependencies, data pull (dvc pull)
## Running It            — training, inference, evaluation commands
## Results               — headline metrics vs baseline
## Extending              — how to swap backbones/datasets/configs
## References             — papers, related repos
```

### Experiment Documentation Practices

```
EXPERIMENT LOG ENTRY (per run)
──────────────────────────────────────────────────────────────────────────
┌─────────────────────────────────────────────────────────────────────────┐
│ Experiment ID:   exp_2026_014                                           │
│ Hypothesis:      Scale-invariant loss will reduce AbsRel vs plain MSE  │
│ Data version:    data/nyu_kitti_merged.dvc @ 9f31ab2                   │
│ Config:          configs/depth_vit_scaleinv.yaml                       │
│ Result:          AbsRel 0.18 → 0.14 (PASS — merged to develop)         │
│ Tracking:        MLflow run 7a2f9c / W&B run xk3f9d                    │
└─────────────────────────────────────────────────────────────────────────┘

TOOLS THAT AUTOMATE THIS
──────────────────────────────────────────────────────────────────────────
• MLflow Model Registry — versioned models + stage transitions (staging→prod)
• Weights & Biases Reports — shareable write-ups with embedded plots
• DVC metrics/plots — lightweight, Git-native alternative
```

---

## 6. Course Project: Standardize Your ML Project

### Goal

Take an existing academic or personal ML/CV project (yours) and bring it up
to the standard covered in this class: versioned code, versioned data,
reviewable structure, and documentation a stranger could onboard from.

**Don't have a project ready?** Use this repo's `lab/` folder (the ViT
depth estimation lab) instead — it's already structured as
`src/{config,model,dataset,train,evaluate,infer}.py` with a synthetic
dataset generator, so you can go straight to Steps 3-6 below: DVC-track
`data/synthetic_depth/` (see `lab/DVC_PRACTICE.md`), add a model card, and
open a self-reviewed PR.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BEFORE / AFTER STANDARDIZATION                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  BEFORE                              │  AFTER                               │
│  ──────                              │  ─────                               │
│  data_final_v2_USE_THIS.zip          │  DVC-tracked data/ with .dvc pointer │
│  one_notebook_does_everything.ipynb  │  src/{data,model,train,eval}.py      │
│  No README                           │  README with setup + results         │
│  main branch only, no PRs            │  develop/exp branches + PR review    │
│  "the model" (no version, no docs)   │  model card + registered version     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Guide

```
STANDARDIZATION WALKTHROUGH
──────────────────────────────────────────────────────────────────────────

STEP 1: Audit
  ├── List every file that currently holds "state": data, weights, configs
  ├── Note what's currently undocumented (env? how to run? what dataset?)
  └── Identify what's in Git today that shouldn't be (large files)

STEP 2: Git hygiene
  ├── git init (if needed), add .gitignore (Section 2 starter)
  ├── nbstripout --install if notebooks are involved
  └── Adopt branch naming: main / develop / exp/*, feature/*

STEP 3: Data versioning
  ├── dvc init, configure a remote (even a local folder works to start)
  ├── dvc add on your dataset(s) and model weight files
  └── Commit the .dvc pointer files to Git, dvc push the data

STEP 4: Restructure code
  ├── Break the "one notebook" into src/{data.py, model.py, train.py, eval.py}
  └── Move hyperparameters into a config file (yaml/json), not hardcoded

STEP 5: Document
  ├── Write a README (Section 5 skeleton)
  └── Write a model card for your best checkpoint (Section 5 template)

STEP 6: Review
  └── Open a PR (even to yourself) using the PR template; run the checklist
```

### Deliverables for This Project

```
✓ Git repo with .gitignore and at least one exp/* branch merged via PR
✓ DVC-tracked dataset and model weights, pushed to a remote
✓ dvc.yaml pipeline with at least 2 stages (e.g. prepare → train)
✓ README.md following the skeleton in Section 5
✓ One completed model card for your best model
✓ Self-reviewed PR using the checklist in Section 4
✓ Stretch: a `.github/workflows/ci.yml` that lints + smoke-trains on PRs
  (see Section 4's "CI/CD with GitHub Actions for ML")
```

---

## 7. Tooling Cheat Sheet

```
┌────────────────────┬────────────────────────────────────────────────────┐
│ Task               │ Command / Tool                                     │
├────────────────────┼────────────────────────────────────────────────────┤
│ New experiment      │ git checkout -b exp/<hypothesis>                  │
│ branch                                                                    │
├────────────────────┼────────────────────────────────────────────────────┤
│ Track a dataset     │ dvc add data/ && git add data.dvc && dvc push     │
├────────────────────┼────────────────────────────────────────────────────┤
│ Reproduce pipeline  │ dvc repro                                          │
├────────────────────┼────────────────────────────────────────────────────┤
│ Compare experiments │ dvc exp show / dvc metrics diff                   │
├────────────────────┼────────────────────────────────────────────────────┤
│ Strip notebook      │ nbstripout --install                              │
│ outputs on commit                                                        │
├────────────────────┼────────────────────────────────────────────────────┤
│ Lint + format       │ ruff check . && black .                           │
├────────────────────┼────────────────────────────────────────────────────┤
│ Pre-commit hooks    │ pre-commit install                                │
├────────────────────┼────────────────────────────────────────────────────┤
│ Log a model version │ MLflow Model Registry / W&B Artifacts             │
├────────────────────┼────────────────────────────────────────────────────┤
│ Check out a PR      │ git fetch origin pull/<n>/head:review && git      │
│ locally             │ checkout review                                   │
├────────────────────┼────────────────────────────────────────────────────┤
│ Diff a branch vs    │ git diff main...exp/<branch>                      │
│ its base only                                                            │
├────────────────────┼────────────────────────────────────────────────────┤
│ Squash-merge a PR   │ GitHub UI: "Squash and merge" (or                 │
│                     │ git merge --squash exp/<branch>)                  │
└────────────────────┴────────────────────────────────────────────────────┘
```

---

## 8. Course Discussion

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DISCUSSION TOPICS FOR CLASS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. TOOLING EXPERIENCE                                                      │
│     • Have you used Git branching in a team setting before?                │
│     • Have you ever lost work or reproducibility because of untracked      │
│       data or an unversioned model?                                        │
│                                                                             │
│  2. YOUR PROJECT                                                            │
│     • What's the current state of the project you'll standardize?          │
│     • What's the messiest part you expect to clean up?                     │
│                                                                             │
│  3. REVIEW CULTURE                                                          │
│     • Has anyone reviewed your ML code before? What did they catch?        │
│     • What would you want a reviewer to catch in your own code?            │
│                                                                             │
│  4. DOCUMENTATION                                                           │
│     • Have you ever revisited your own old project and not understood it?  │
│     • What's the one thing you wish past-you had written down?             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Homework / Next Steps

```
HOMEWORK BEFORE NEXT CLASS (Data Pipeline in Industry)
═══════════════════════════════════════════════════════════════

1. STANDARDIZE YOUR PROJECT (see Section 6 deliverables)
   ├── Git repo with proper .gitignore and branch structure
   ├── DVC-tracked data + at least a 2-stage dvc.yaml pipeline
   ├── README.md + one model card
   ├── No project of your own? Use `lab/` in this repo (ViT depth lab) --
   │   run through `lab/DVC_PRACTICE.md` for the DVC steps
   └── Push everything and share the repo link

2. GIT/DVC/CI PRACTICE
   ├── Practice: open a PR, self-review it with the checklist
   ├── Practice: `git fetch origin pull/<n>/head:review` a real PR (yours
   │   or a classmate's) and run it locally instead of only reading the diff
   ├── Practice: dvc exp run with 2+ different hyperparameter values
   │   (try it on `lab/params.yaml` -- e.g. train.lr or train.loss_fn)
   └── Stretch: add a `.github/workflows/ci.yml` that lints and smoke-trains
       on every PR (see Section 4)

3. READING FOR CLASS 3
   ├── Skim: "Designing Machine Learning Systems" ch. on data pipelines
   ├── Browse: how a company you admire structures its data pipeline
   │   (e.g. Uber Michelangelo, Netflix Metaflow blog posts)
   └── Think about: what does a production data pipeline for IMAGE data
       need that one for tabular data doesn't?

RESOURCES:
─────────────────
• DVC docs: https://dvc.org/doc
• Model Cards paper: https://arxiv.org/abs/1810.03993
• Conventional Commits: https://www.conventionalcommits.org
• GitHub Actions docs: https://docs.github.com/actions
• CML (metrics-on-PR): https://cml.dev
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      KEY TAKEAWAYS FROM CLASS 2                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ✓ Git handles code; DVC handles data and model artifacts — use both       │
│                                                                             │
│  ✓ Experiment branches are expected to fail; that's a feature, not a bug   │
│                                                                             │
│  ✓ ML code review must catch data leakage and reproducibility issues,      │
│    not just style and logic bugs                                          │
│                                                                             │
│  ✓ Model cards make a model's intended use, performance, and limitations   │
│    explicit — for teammates, downstream users, and future you             │
│                                                                             │
│  ✓ Reviewing "via git" means running a teammate's branch locally, not      │
│    just reading a diff — branch protection + CI turn the checklist into   │
│    something enforced, not just remembered                                │
│                                                                             │
│  ✓ GitHub Actions CI/CD for ML adds two things standard software CI       │
│    doesn't need: pulling DVC-tracked data, and a smoke-train step         │
│                                                                             │
│  ✓ Good tooling turns a personal experiment into team-reviewable,          │
│    production-ready work                                                  │
│                                                                             │
│  ✓ Next class: how production data pipelines are built for both           │
│    structured and unstructured (image/video) data                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Contact & Questions

**Instructor**: [Your Name]
**Email**: [Your Email]
**Office Hours**: [Schedule]
**Course Repo**: [GitHub Link]

---

*This document was created for educational purposes. Feel free to share and adapt with attribution.*

**Last Updated**: [Date]
**Version**: 1.0
