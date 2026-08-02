# Optimizing DeblurGAN-v2 for Motion Blur Removal in Low-Resource Environments Using Structured Pruning

MSc in Advanced Software Engineering, Final Project
Shayan Wickrama Arachchi | IIT ID: 20240429 | UoW ID: w21067752
Supervisor: Mr. Yasin Miran

## Purpose

This project investigates how structured (channel-level) pruning affects the
deblurring quality and computational characteristics of the FPN generator in
DeblurGAN-v2. L1-norm soft pruning is applied at 10%, 30% and 50% ratios.
Each pruned variant is then fine-tuned for 50 epochs on the GoPro training
set with the original composite loss (pixel MSE + VGG-19 perceptual +
relativistic adversarial), and all variants are evaluated on the full
official GoPro test set (1,111 images) using PSNR, SSIM, inference latency,
peak GPU memory and model size.

## Repository contents

| Path | Description |
|---|---|
| `deblurgan_v2_full_experiment.ipynb` | The complete experiment: setup, dataset pipeline, baseline evaluation, multi-ratio pruning, fine-tuning, evaluation, figures, and post-hoc diagnostics. This single notebook produces every number and figure in the report. |
| `quick_demo.ipynb` | A short demo that loads the trained models from Drive and deblurs a few images, including one you upload yourself. No training involved, runs in a few minutes on a GPU runtime. Needs the checkpoints saved by the main notebook. |
| `results/` | Result files produced by the experiment: per-ratio metrics (`ratio_*/results.json`), fine-tuning histories (`ratio_*/history.json`), the consolidated `experiment_results.json`, baseline metrics, the clean-session measurements (`results_clean_full_train.json`, the source of every number in the report), and the report figures. |
| `results/deblurgan_v2_full_experiment_executed.ipynb` | An executed copy of the notebook with all cell outputs from the actual run, kept as evidence of the training and evaluation logs. |

## Software and hardware requirements

- Google Colab with a GPU runtime (Runtime > Change runtime type > GPU).
  Any modern Colab GPU works. Fine-tuning takes roughly 3 to 5 minutes per
  epoch depending on the GPU assigned, which works out to about 3 hours per
  pruning ratio and around 10 hours for the full experiment.
- Google Drive with approximately 15 GB free. The dataset zip is ~8.8 GB,
  the model weights ~244 MB, and training checkpoints take ~750 MB per
  in-progress ratio.
- No local installation is required or supported. The notebook is designed
  to run end-to-end on Colab.

## Languages, libraries and frameworks

Python 3 (Colab runtime), PyTorch 2.x with CUDA, torchvision (VGG-19 for the
perceptual loss), pretrainedmodels 0.7.4 (Inception-ResNet-v2 backbone),
albumentations 1.3.0, scikit-image (PSNR/SSIM), OpenCV, NumPy, matplotlib and
tqdm. The generator architecture comes from the official
[VITA-Group/DeblurGANv2](https://github.com/VITA-Group/DeblurGANv2)
repository (BSD-licensed), which the notebook clones automatically at
runtime.

All dependencies are installed by the first cells of the notebook itself, so
there is no separate requirements file.

## One-time setup (before the first run)

1. Pre-trained generator weights: download `fpn_inception.h5` (~244 MB) from
   the pre-trained models section of the official DeblurGANv2 repository
   (https://github.com/VITA-Group/DeblurGANv2) and place it at the root of
   your Google Drive (`MyDrive/fpn_inception.h5`).
2. GoPro dataset (full official split): download `GOPRO_Large.zip` (~8.8 GB)
   from https://seungjunnah.github.io/Datasets/gopro and upload the zip as-is
   to Drive at `MyDrive/datasets/GOPRO_Large.zip`. Do not unzip it on Drive.
   Each session the notebook copies the zip to the runtime's local disk and
   extracts it there, because training over the Drive mount is far too slow.
   The notebook asserts the official pair counts (2,103 train / 1,111 test)
   after extraction.

## Configuration

All paths and hyperparameters live in the Configuration cell (Section 2 of
the notebook): Drive paths, pruning ratios, epochs, batch size, learning
rates, loss weights and validation cadence. The defaults reproduce the
experiment reported in the thesis, so nothing needs to be changed to re-run
it.

## Running the experiment

1. Open `deblurgan_v2_full_experiment.ipynb` in Google Colab.
2. Select a GPU runtime.
3. Run all cells (Runtime > Run all). The notebook mounts Google Drive
   (authorisation prompt on first run), installs dependencies, prepares the
   dataset, evaluates the baseline, and runs the prune, fine-tune and
   evaluate cycle for each pruning ratio in turn.

The experiment is built around Colab's session limits:

- A checkpoint is written to Drive after every epoch. If the session
  disconnects, simply Run All again: completed ratios are skipped entirely
  and the in-progress ratio resumes from its last completed epoch.
- Expensive evaluations (baseline, pre-fine-tuning) are cached to Drive as
  JSON and reused on re-runs.
- The generator with the best validation PSNR is checkpointed separately
  (`checkpoint_best.pth`) and is the one used for final evaluation.

Results are written to Drive under `checkpoints/deblurgan_pruning_full_v2/`
(the consolidated `experiment_results.json`, per-ratio `results.json` and
`history.json`, and final model weights) and figures to `MyDrive/figures/`.
Copies of the result JSONs and figures are included in `results/` in this
submission.

## Evaluating or reproducing individual parts

- Baseline evaluation only: run Sections 1 to 6. The baseline metrics load
  from cache if present, or are computed on the full test set in about 15
  minutes.
- Diagnostics (Section 9): two post-hoc checks used in the evaluation
  chapter of the report. Diagnostic A (about 3 minutes) compares eval-mode
  and train-mode inference to explain the gap between the reproduced
  baseline and the published benchmark. Diagnostic B (about 50 minutes, run
  in a fresh session after the main experiment) re-measures all seven models
  (the baseline, the three pruned variants before fine-tuning and the three
  after) in a single clean session, and stores the per-image PSNR/SSIM values
  used for the Wilcoxon significance tests. All quality numbers in the report
  come from this measurement.
- Trained model files are not included in this submission due to their size
  (244 MB each). They are reproducible by running the notebook, and the
  exact metrics they produce are recorded in `results/`.

## Known limitations

- Soft pruning: channels are zeroed rather than physically removed (a
  workaround for torch-pruning's dependency-tracing failure on the Instance
  Normalisation layers in the FPN), so file size, latency and memory do not
  decrease. The study reports effective (non-zero) parameter counts and
  theoretical hard-pruned sizes instead, and positions hard pruning as
  future work.
- Evaluation protocol matters: the official inference script runs the
  network in training mode, and under that protocol the pre-trained
  checkpoint reproduces the published benchmark (29.15 dB vs 29.55 dB).
  Under conventional eval-mode inference the same weights score several dB
  lower, because the Inception backbone's BatchNorm running statistics only
  apply in eval mode (Diagnostic A). Absolute scores also vary between Colab
  sessions (GPU class and software stack), so all reported numbers come from
  a single clean session in train mode (Diagnostic B).
- Discriminator: the pre-trained discriminator is not distributed with
  DeblurGAN-v2, so a patch-only PatchGAN with the relativistic average loss
  is trained from scratch during fine-tuning. The original paper uses a
  double-scale design.
- The GPU type assigned by Colab affects wall-clock time only. Quality
  metrics are deterministic given the fixed pruning masks.

## External services

Google Colab and Google Drive only. No API keys, credentials or paid
services are required.
