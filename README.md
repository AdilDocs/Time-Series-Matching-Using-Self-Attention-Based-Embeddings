# Source Code for "A Self-Attention Framework for Time Series Matching"

This folder contains the complete, self-contained Python implementation
used to produce every table and figure reported in the paper. No
external machine learning or deep learning framework is used: the
self-attention encoder, its automatic differentiation, and the training
procedure are implemented directly in NumPy, so that the full
computational pipeline is transparent and auditable end to end.

## Requirements

```
numpy
matplotlib
pandas
scikit-learn   (used only for PCA/t-SNE in the embedding-visualization figures)
numba          (used only by step16 to compile DTW; numerically identical to step1)
```

## Data

Experiments use the UCR Time Series Classification Archive (2015
edition). This is a standard, publicly available benchmark and is not
included in this package; it can be obtained from the archive's official
distribution. Once obtained, place it alongside these scripts as:

```
UCR_TS_Archive_2015/

```

for each of the 20 datasets used in the paper: Coffee, ItalyPowerDemand,
Beef, Lighting7, Gun_Point, ECG200, Trace, OSULeaf, StarLightCurves,
BirdChicken, DiatomSizeReduction, ArrowHead, ECGFiveDays, Computers,
Earthquakes, CBF, CinC_ECG_torso, Adiac, OliveOil, and Wafer.

## Code Structure

### Core pipeline

| File | Contents |
|---|---|
| `step1.py` | Data loading, PAA downsampling, Euclidean and DTW baseline classifiers |
| `step2.py` | Reverse-mode automatic differentiation engine (implemented from scratch) |
| `step3.py` | Self-attention encoder architecture (Sections 3.1-3.2 of the paper) |
| `step4.py` | Training procedure: triplet margin loss with Adam optimization (Section 3.3) |
| `step5.py` | Single-dataset training/evaluation entry point |
| `step6.py` | Multi-dataset experiment driver, including the initialization-collapse detection and correction procedure discussed in the paper |

Each of `step1.py`-`step4.py` includes a self-contained correctness check
runnable directly (`python3 step2.py`, etc.), verifying gradient
correctness against numerical finite differences and reproducing the
convergence behavior reported during development.

### Scripts producing specific paper results

| File | Paper result | Output |
|---|---|---|
| `step7_compile_results.py` | Table I (accuracy) and Table II (response time) | `results_compiled.csv` |
| `step8_fig_error_comparison.py` | Error-comparison figure | `fig6_error_comparison.png` |
| `step9_fig_time_comparison.py` | Response-time figure | `fig7_time_comparison.png` |
| `step10_fig_k_sensitivity.py` | Accuracy-versus-k figure | `fig9_accuracy_vs_k.png` |
| `step16_knn_fair_comparison.py` | Table III (Att/ED/DTW with k=1 vs. validation-selected k) | `knn_fair_comparison.csv` |
| `step11_dmodel_tradeoff_sweep.py` | Table IV (embedding-dimension trade-off) | `tradeoff_results.csv` |
| `step12_fig_dmodel_tradeoff.py` | Embedding-dimension trade-off figure (values from step11) | `fig9b_dmodel_tradeoff.png` |
| `step17_gunpoint_dmodel.py` | Gun_Point d sweep cited with Table IV | `tradeoff_gunpoint.csv` |
| `step13_train_with_snapshots.py` | Checkpoints for the embedding-evolution analysis | `checkpoints_evolution/` |
| `step14c_checkpoint_accuracy.py` | Table V (k and test accuracy per checkpoint) | `checkpoint_accuracy.csv` |
| `step14_fig_embedding_evolution.py` | PCA/t-SNE embedding-evolution figures | `embedding_evolution_*_combined_full.png`, `*_zoomed.png` |
| `step14b_fig_loss_curve.py` | Training-loss figure (loss at each saved checkpoint) | `embedding_evolution_*_loss_full.png` |
| `step15_case_study_attention.py` | Gun_Point case-study attention figure | `case_study_gun_point_attention.png` |
| `run_all_parallel.py` | Optional: runs `step6.run_dataset` for all 20 datasets on 2 CPU cores | `step6_run_results.csv`, `logs/` |

All result files, figures, and trained checkpoints from the run reported
in the revised paper are included in this folder, so every table and
figure can be regenerated without retraining (`checkpoints/`,
`checkpoints_tradeoff/`, `checkpoints_evolution/`).

## Reproducing the Results

1. Place the UCR archive as described above. The 2015 archive stores
   Wafer in lowercase (`wafer/wafer_TRAIN`); create a folder `Wafer/`
   containing `Wafer_TRAIN` and `Wafer_TEST` (copies or links of the
   lowercase files).
2. Train and evaluate all 20 datasets (`python3 step6.py`, option 1, or
   `python3 run_all_parallel.py` on a 2-core machine). One checkpoint per
   dataset is saved under `checkpoints/`; existing checkpoints are
   resumed rather than retrained.
3. Tables I-III and their figures:
   ```
   python3 step7_compile_results.py
   python3 step8_fig_error_comparison.py
   python3 step9_fig_time_comparison.py
   python3 step10_fig_k_sensitivity.py
   python3 step16_knn_fair_comparison.py
   ```
4. Table IV and its figure: `step11`, `step12`, `step17`.
5. Table V and the embedding-evolution figures: `step13`, `step14c`,
   `step14`, `step14b`. `step13` resets Adam's moment estimates at every
   checkpoint from iteration 40 onward; this schedule reproduces the
   paper's checkpoints exactly.
6. Case study: `step15`.

Response times depend on the machine; the reported times were measured
on a single core of a 2.1 GHz Intel Xeon processor. `step11` and
`step13` retrain at multiple settings or checkpoints and take longer
than the other scripts.
