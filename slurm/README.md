# SLURM Scripts for HPC

Submit jobs from project root (e.g. `/scratch/jw310/adni-dttvit`).

## Prerequisites

```bash
mkdir -p runs/logs results
```

- Log dir: `runs/logs`
- Results dir: `results`
- Activate venv: `/scratch/jw310/venvs/torch_fix`

## Scripts

| Script | Job | Time | Description |
|--------|-----|------|-------------|
| `baseline.sbatch` | vit2d_baseline | 12h | Full ViT baseline, single-slice fixed z |
| `baseline_25d.sbatch` | vit2d_baseline_25d | 12h | Same protocol as baseline, 2.5D `stack3`/`stack5` (env `Z_INDEX`, `SLICE_STACK_MODE`) |
| `baseline_25d_array_z.sbatch` | vit2d_25d_z | 12h | **Array** `z ∈ {77,115,127,144}` in parallel (one GPU each); optional `SLICE_STACK_MODE` |
| `slice_selection.sbatch` | slice_sel | 24h | Find best z (19,48,77,115,144,173) |
| `compare_thin_methods.sbatch` | thin_compare | 48h | Baseline + DTT+EE (l2, attn, random) |
| `dtt_ee.sbatch` | vit2d_dtt_ee | 12h | Single DTT+EE run (default attn) |
| `teacher_student.sbatch` | teacher_stu | 48h | Two-stage: teacher → student distill |
| `teacher_student_skip.sbatch` | teacher_stu2 | 24h | Student only (teacher already trained) |
| `eval_early_exit_tau.sbatch` | eval_tau | 2h | Evaluate early-exit at different tau |
| `compare_ablation.sbatch` | ablation | 36h | Baseline, Only EE, +DTT, +DTT+EE |
| `compare_dtt_only.sbatch` | dtt_only | 24h | Stage 1: DTT(L2/Attn/Random) only |
| `eval_compute_metrics.sbatch` | eval_metrics | 2h | Accuracy + time + speedup for all configs |

## Submit

```bash
cd /scratch/jw310/adni-dttvit

# Baseline
sbatch slurm/active/baseline.sbatch

# 2.5D baseline (default stack3, z=144; override e.g. Z_INDEX=127)
# Z_INDEX=127 SLICE_STACK_MODE=stack3 sbatch slurm/active/baseline_25d.sbatch
sbatch slurm/active/baseline_25d.sbatch

# 2.5D for z in 77,115,127,144 at once (4 GPUs if scheduler allows)
# SLICE_STACK_MODE=stack5 sbatch slurm/active/baseline_25d_array_z.sbatch
sbatch slurm/active/baseline_25d_array_z.sbatch

# Slice selection (find best z)
sbatch slurm/active/slice_selection.sbatch

# Compare thin methods (l2, attn, random)
sbatch slurm/active/compare_thin_methods.sbatch

# DTT + Early Exit (single method)
sbatch slurm/active/dtt_ee.sbatch

# DTT + EE with specific method
THIN_METHOD=l2 sbatch slurm/active/dtt_ee.sbatch

# Teacher-Student (full)
sbatch slurm/active/teacher_student.sbatch

# Teacher-Student (student only)
sbatch slurm/active/teacher_student_skip.sbatch

# Early-exit tau evaluation
sbatch --export=CKPT=runs/vit2d_dtt_ee_attn/best.pt slurm/active/eval_early_exit_tau.sbatch

# Ablation (Baseline / Only EE / +DTT / +DTT+EE)
sbatch slurm/active/compare_ablation.sbatch

# Stage 1: DTT only (L2, Attn, Random)
sbatch slurm/active/compare_dtt_only.sbatch

# Compute metrics (accuracy + time + speedup)
sbatch slurm/active/eval_compute_metrics.sbatch
```

## Customize

Edit `PROJECT` and `VENV` at the top of each script if your paths differ.
