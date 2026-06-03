# Repository Guidelines

## Project Structure & Module Organization

This repository is organized as a staged learning and experiment project for flow matching and diffusion models.

- `src/`: reusable Python modules, including ODE/SDE solvers, toy data, MLPs, samplers, diffusion utilities, UNet, and DiT components.
- `labs/`: runnable experiment scripts grouped by stage, for example `labs/lab1/`, `labs/lab2/`, `labs/lab_cifar_flow/`, and `labs/lab_dit/`.
- `notes/` and `reports/`: theory notes, experiment writeups, and summaries.
- `configs/`: YAML configs for larger image experiments.
- `figures/`, `results/`, `logs/`, `checkpoints/`: generated experiment artifacts. Avoid committing large or incidental outputs unless they are part of a report.
- `notebooks/`: exploratory notebooks matching the staged curriculum.

## Build, Test, and Development Commands

Install dependencies in a Python environment:

```bash
pip install -r requirements.txt
```

Run Stage 1 visualizations and sanity checks:

```bash
python labs/lab1/run_probability_path.py
python labs/lab1/run_ode_sde.py
```

Run the 2D Flow Matching experiment:

```bash
python labs/lab2/run_2d_flow_matching.py --device cuda --train_steps 10000 --batch_size 1024
```

Use CPU smoke runs for quick validation:

```bash
python labs/lab2/run_2d_flow_matching.py --device cpu --train_steps 1000 --batch_size 512
```

Check Python syntax before committing:

```bash
python -m py_compile src/*.py labs/lab1/*.py labs/lab2/*.py
```

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation and clear, typed function signatures where practical. Keep reusable logic in `src/` and keep `labs/` scripts as orchestration layers. Prefer descriptive names such as `sample_ode_euler`, `flow_matching_loss`, and `run_2d_flow_matching.py`. Generated figures should use stable, stage-prefixed names like `figures/stage2/fm_training_loss.png`.

## Testing Guidelines

There is no formal test suite yet. Treat fast smoke runs and numerical sanity checks as the current validation standard. For solver changes, add analytic checks where possible, such as Brownian variance `Var[W_t] ~= t` or ODE comparisons against closed-form solutions. For training scripts, verify that checkpoints, loss curves, and expected figures are produced.

## Commit & Pull Request Guidelines

Existing commits use concise stage or phase prefixes, for example `stage4: ...`, `phase3: ...`, and `stage1 and 2 fix`. Follow that pattern:

```text
stage2: add fixed-noise nfe comparison
phase3: update cifar flow sampling tools
```

Pull requests should include a short description, commands run, key metrics or generated artifact paths, and screenshots when figures or sample quality change. Mention any large outputs intentionally added to `figures/`, `results/`, or `checkpoints/`.

## Agent-Specific Instructions

Do not overwrite unrelated experiment outputs or dirty worktree changes. Before editing generated artifacts, confirm they are required for the current task. Prefer small, reproducible script changes over notebook-only edits.


# Project instructions for Codex

## Language
- 默认使用中文回答。
- 回答要简洁、严谨，像作业答案一样组织。

## Project workflow
- 修改代码前，先阅读 README、配置文件、入口脚本。
- 不要随意删除已有实验日志、checkpoint、数据文件。
- 新增实验时，记录实验目的、运行命令、关键参数、输出路径。
- 遇到报错时，先定位 traceback，再给出最小修改方案。

## Coding rules
- 优先做最小可行修改，不要无理由大规模重构。
- 修改训练代码后，优先运行最小验证脚本。
- 给出结论时区分：已验证事实、合理推测、下一步建议。

## Principle
- 我们一切的原则都是使得我们的模型具有更好的生成能力，在这个基础上(完成这任务的前提下)，我们做一些有价值的消融实验