# On the Limitation of Diffusion-Based Speech Enhancement Models and an Adaptive Selection Strategy

This repository contains the official implementation for my Master's thesis, focusing on an adaptive, two-stage recommender system to select the optimal speech enhancement model for a given noisy audio sample.

## Overview

Modern speech enhancement (SE) models, particularly those based on diffusion, exhibit varying performance across different acoustic conditions. No single model universally outperforms others. This project introduces a novel **two-stage recommender system** designed to analyze a noisy speech signal and predict which SE model (`SGMSE`, `CDiffuSE`, `StoRM`, or `none`) will yield the highest quality enhancement.

The core challenge in training such a recommender is the severe class imbalance, where one model (e.g., `SGMSE`) is often the "best" choice. To overcome this, we designed a two-stage pipeline:

1.  **Stage 1: The Gatekeeper:** A binary classifier that makes the coarse decision of whether the dominant `SGMSE` model is the best choice or not. This model is an **Audio Spectrogram Transformer (AST)** trained only on spectrogram features.
2.  **Stage 2: The Expert:** A fine-grained 3-class classifier that activates only when the Gatekeeper rules out `SGMSE`. This model, a **hybrid AST**, uses both spectrogram and engineered spectral features (CE/KL) to distinguish between the minority classes (`CDiffuSE`, `StoRM`, `All Failed`).

This "divide and conquer" approach was proven to be more effective than a single 4-class model, achieving a robust, cross-validated Macro F1 score of **~0.58-0.60**.

## Acquiring Baseline Models

The recommender system is trained to select from among three state-of-the-art speech enhancement models. Before training the recommender or running the full enhancement pipeline, please acquire the model weights and implementations from their official GitHub repositories:

1.  **SGMSE:** [https://github.com/sp-uhh/sgmse.git](https://github.com/sp-uhh/sgmse.git)
2.  **CDiffuSE:** [https://github.com/neillu23/CDiffuSE.git](https://github.com/neillu23/CDiffuSE.git)
3.  **StoRM:** [https://github.com/sp-uhh/storm.git](https://github.com/sp-uhh/storm.git)

Please follow the setup instructions in each repository to ensure the models are runnable.

## Data Preparation

The training and evaluation data for the recommender was created by mixing clean speech and noise from several public datasets.

### 1. Data Sources

*   **Clean Speech:**
    *   [CREMA-D](https://www.kaggle.com/datasets/ejlok1/cremad)
    *   [LJSpeech](https://www.kaggle.com/datasets/mathurinache/the-lj-speech-dataset)
*   **Noise Signal:**
    *   [MS-SNSD](https://github.com/microsoft/MS-SNSD.git)
*   **Real world recordings**
    *   [DNS Challenge 2020](https://github.com/microsoft/DNS-Challenge)
*   **Benkmark dataset**
    * [Voicebank-DEMAND](https://www.kaggle.com/datasets/jiangwq666/voicebank-demand)

For clean speech and noise signal, please synthesize noisy speech with following command:
```bash
python preprocessing/speech_noise_mixing.py
```

### Label Generation

The ground-truth labels for the recommender system (i.e., which SE model is "best" for a given noisy sample) were generated through a comprehensive, data-driven pipeline. Instead of relying on a single metric, we developed a methodology to categorize the *success* or *failure* of each enhancement process.

The process is as follows:

1.  **Systematic Enhancement:** Every noisy audio sample in our prepared datasets was processed by each of the three baseline enhancement models (`SGMSE`, `CDiffuSE`, `StoRM`).

2.  **Objective Quality Evaluation:** The output of each enhancement was evaluated using the industry-standard **DNSMOS P.835** framework. This provides a multi-dimensional Mean Opinion Score (MOS) predicting human listener ratings on three independent scales:
    *   **SIG (Speech Quality):** The clarity and lack of distortion in the speech signal itself.
    *   **BAK (Background Noise Quality):** The degree of suppression and lack of annoying artifacts in the residual background.
    *   **OVR (Overall Quality):** The overall impression of the audio quality.

3.  **Determining the "Best" Model:** For each noisy sample, the enhancement model that achieved the highest **DNSMOS OVR (Overall) score** was designated as the ground-truth "best model" for that sample.

4.  **Handling Ambiguity ("All Failed"):** In cases where no enhancement model was able to produce a satisfactory result (e.g., all models scored below a certain quality threshold on DNSMOS OVR), the sample was labeled as `All Failed`. This created a crucial fourth category for the recommender, allowing it to learn to identify situations where no available model is likely to succeed.

This entire process is automated in the `enhancement_driver.py` script (or a similar name based on your file). The script includes a `SpeechEnhancementPipeline` to run the baseline models and a `QualityEvaluation` class to score the outputs and generate the final label CSV files.


### 3. Feature Pre-calculation (Optional but Recommended)

For the hybrid Expert model, training can be significantly accelerated by pre-calculating the CE/KL features.

```bash
python preprocess_features.py \
    --base_data_dir /path/to/your/datasets \
    --output_base_dir /path/to/save/cekl_features/
```

## Training the Two-Stage Recommender

The recommender is trained in two separate stages using the `model_recommender_w_scaler.py` script. All stages use a pre-trained `MIT/ast-finetuned-audioset-10-10-0.4593` model from the Hugging Face Hub.

### Stage 1: Training the Gatekeeper

This trains a binary classifier (`SGMSE` vs. `NOT_SGMSE`).

```bash
python model_recommender_w_scaler.py \
    --feature_type mel_spec \
    --pretrained_models ast_finetuned_audioset \
    --training_stage gatekeeper \
    --use_weighted_sampler \
    --learning_rate 2.5e-5 \
    --weight_decay 1e-4 \
    --batch_size 32 \
    --num_epochs 50 \
    --gatekeeper_model_path "./gatekeeper_model_fold_{fold}_best.pth"
```

### Stage 2: Training the Expert

This trains a 3-class classifier on a filtered dataset (excluding `SGMSE` samples). We recommend using the hybrid model for this stage.

```bash
python model_recommender_w_scaler.py \
    --feature_type mel_spec \
    --pretrained_models ast_finetuned_audioset \
    --training_stage expert \
    --use_weighted_sampler \
    --use_hybrid_features \
    --cekl_features_base_dir /path/to/save/cekl_features/ \
    --learning_rate 2.5e-5 \
    --weight_decay 1e-4 \
    --batch_size 32 \
    --num_epochs 50 \
    --expert_model_path "./expert_model_fold_1_best_hybrid.pth"
```

## Evaluation

Once both the Gatekeeper and Expert models are trained for each fold, you can evaluate the performance of the full two-stage pipeline.

### Evaluating the Pipeline Performance

This command runs the full pipeline on a validation set and performs the detailed bottleneck analysis.

```bash
python model_recommender_w_scaler.py \
    --feature_type mel_spec \
    --pretrained_models ast_finetuned_audioset \
    --training_stage evaluate_pipeline \
    --use_hybrid_features \
    --cekl_features_base_dir /path/to/save/cekl_features/ \
    --gatekeeper_model_path "./gatekeeper_model_fold_{fold}_best.pth" \
    --expert_model_path "./expert_model_fold_1_best_hybrid.pth"
```

### Running the Full Recommendation & Enhancement System

To use the trained two-stage pipeline to enhance a directory of noisy audio files, use the `recommendation_driver.py` script.

```bash
python recommendation_driver.py \
    --noisy_speech_dir /path/to/new_noisy_audio \
    --enhanced_dir /path/to/output_directory \
    --gatekeeper_model_path /path/to/your/best_gatekeeper.pth \
    --expert_model_path /path/to/your/best_expert.pth \
    --use_hybrid_expert \
    --cekl_features_base_dir /path/to/precalculated/cekl_features
```
This will create subdirectories in the output folder (`rec_SGMSE`, `rec_CDiffuSE`, etc.), populate them based on the recommender's predictions, and then run the corresponding speech enhancement models.

### Recommendation-Driven Enhancement & Evaluation

This project culminates in the `recommendation_driver.py` script, a comprehensive tool to apply the trained recommender system to new noisy audio and evaluate the quality of the resulting adaptively enhanced speech.

#### Overview of the Pipeline

The script performs the following steps:

1.  **Load Recommender:** It initializes the best-performing two-stage pipeline, loading the pre-trained **Gatekeeper** (AST-Hybrid model) and **Expert** (AST-Hybrid model).
2.  **Make Recommendations:** It iterates through a directory of unseen noisy audio files. For each file, it performs the two-stage prediction to determine the optimal speech enhancement model (`SGMSE`, `CDiffuSE`, `StoRM`, or `All Failed`).
3.  **Sort Audio Files:** Based on the recommendations, it creates subdirectories (e.g., `rec_SGMSE`, `rec_CDiffuSE`) and copies the noisy audio files into the appropriate folder. This prepares the data for targeted enhancement.
4.  **Run Targeted Enhancement:** The script then invokes the `enhancement_driver.py` (your separate implementation) for each subdirectory, running only the recommended SE model on the files sorted into its folder. For files classified as `All Failed`, a default model (`SGMSE`) is used as a fallback.
5.  **Objective Evaluation:** Finally, the script can perform a detailed objective evaluation of the enhanced audio, comparing the outputs against the original noisy (and optionally, clean) speech using standard speech quality metrics:
    *   **PESQ** (Perceptual Evaluation of Speech Quality)
    *   **STOI** (Short-Time Objective Intelligibility)
    *   **DNSMOS P.835** (A deep learning-based MOS predictor for overall quality, signal quality, and background quality)

#### Usage

The script supports two primary modes: a full **recommend-and-enhance** workflow and an **evaluation-only** mode.

1. Full Pipeline: Recommending, Enhancing, and Evaluating

This is the standard workflow for processing a new set of noisy files. You must have the baseline SE models (`SGMSE`, etc.) and your trained recommender models available.

**Command:**
```bash
python recommendation_driver.py \
    --noisy_speech_dir /path/to/your/unseen_noisy_audio \
    --enhanced_dir /path/to/your/output_directory \
    --gatekeeper_model_path /path/to/best_gatekeeper.pth \
    --expert_model_path /path/to/best_hybrid_expert.pth \
    --use_hybrid_features \
    --cekl_features_base_dir /path/to/precalculated/cekl_features \
    --clean_speech_dir /path/to/corresponding_clean_audio # Optional, for PESQ/STOI
```

**Arguments:**
*   `--noisy_speech_dir`: The input directory containing new `.wav` files.
*   `--enhanced_dir`: The main output directory where the `rec_*` subfolders and enhanced audio will be created.
*   `--gatekeeper_model_path`: Path to the trained `.pth` file for your best Gatekeeper model.
*   `--expert_model_path`: Path to the trained `.pth` file for your best (hybrid) Expert model.
*   `--use_hybrid_features`: Flag to enable the hybrid feature path for both models.
*   `--cekl_features_base_dir`: Path to the pre-calculated CE/KL features for the noisy audio.
*   `--clean_speech_dir`: (Optional) If you have corresponding clean reference audio, provide the path to calculate PESQ and STOI. The filenames must match the noisy audio.

2. Evaluation-Only Mode

This mode is useful if you have already run the enhancement pipeline and simply want to re-calculate or analyze the objective scores. It expects the output directory to already be populated with the `rec_*` folders and their corresponding `enhanced` subfolders.

**Command:**
```bash
python recommendation_driver.py \
    --eval_only \
    --enhanced_dir /path/to/your/output_directory \
    --clean_speech_dir /path/to/corresponding_clean_audio # Optional
```

The script will automatically find all `rec_*` folders within the `--enhanced_dir`, locate the noisy and enhanced files for each, and calculate the objective metrics, providing a final, aggregated performance report for the entire adaptively enhanced dataset.

## Key Findings

- A single, end-to-end 4-class model is fundamentally limited by class imbalance, plateauing at a Macro F1 score of ~0.55.
- A two-stage pipeline, which decouples the majority-class decision from the minority-class decision, significantly outperforms the single-model approach.
- The Expert model, when trained on filtered data without the majority class, achieves a near-perfect classification score on its specialized task.
- The performance of the entire system is bottlenecked by the binary Gatekeeper model's ability to handle ambiguous cases at the decision boundary.
- The final, optimized pipeline achieves a cross-validated Macro F1 score of **~0.58-0.60**, demonstrating a robust and effective strategy for adaptive speech enhancement.
