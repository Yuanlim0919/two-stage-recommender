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
    *   [Voicebank-DEMAND](https://www.kaggle.com/datasets/jiangwq666/voicebank-demand)
    *   [DNS Challenge 2020](https://github.com/microsoft/DNS-Challenge)
*   **Noise:**
    *   [MS-SNSD](https://github.com/microsoft/MS-SNSD.git)
    *   [Voicebank-DEMAND](https://www.kaggle.com/datasets/jiangwq666/voicebank-demand)

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

## Key Findings

- A single, end-to-end 4-class model is fundamentally limited by class imbalance, plateauing at a Macro F1 score of ~0.55.
- A two-stage pipeline, which decouples the majority-class decision from the minority-class decision, significantly outperforms the single-model approach.
- The Expert model, when trained on filtered data without the majority class, achieves a near-perfect classification score on its specialized task.
- The performance of the entire system is bottlenecked by the binary Gatekeeper model's ability to handle ambiguous cases at the decision boundary.
- The final, optimized pipeline achieves a cross-validated Macro F1 score of **~0.58-0.60**, demonstrating a robust and effective strategy for adaptive speech enhancement.
