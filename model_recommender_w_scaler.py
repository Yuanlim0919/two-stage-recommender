import argparse
import os
import warnings
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split, Subset, WeightedRandomSampler
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from utils.ce_kl_feature_extraction import NoiseDataPreprocessor, prepare_feature_standard_scaler_by_fold
from transformers import ASTFeatureExtractor, ASTModel
from collections import Counter
import timm
from pathlib import Path

class RecommenderModelMelSpec(torch.nn.Module):
    # Added BatchNorm and Dropout
    def __init__(self, hidden_size, output_size=4, adaptive_pool_output_size=(16, 16), dropout_rate=0.3):
        super(RecommenderModelMelSpec, self).__init__()
        self.conv_layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d(adaptive_pool_output_size)
        input_size_fc = 32 * adaptive_pool_output_size[0] * adaptive_pool_output_size[1]
        self.flatten = torch.nn.Flatten()
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(input_size_fc, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x


class RecommenderModelCeKl(torch.nn.Module):
    # Added BatchNorm and Dropout
    def __init__(self, input_size=9, hidden_size=128, output_size=4, dropout_rate=0.3,):
        super(RecommenderModelCeKl, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_size, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.layers(x)


class RecommenderModelMelSpecPretrain(torch.nn.Module):
    def __init__(self, backbone, num_backbone_features=512, output_size=4,dropout_rate=0.3):
        super(RecommenderModelMelSpecPretrain, self).__init__()
        self.backbone = backbone
        self.num_backbone_features = num_backbone_features
        self.fc_layers = torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_backbone_features),
            torch.nn.Linear(num_backbone_features, num_backbone_features // 4),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(num_backbone_features // 4, output_size)
        )
    
    def forward(self, x):
        # This model is used for pretraining, so it doesn't have a forward method.
        # It will be used to extract features from Mel Spectrograms.
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten the features
        outputs = self.fc_layers(features)
        return outputs    


class RecommenderModelAST(torch.nn.Module):
    def __init__(self, backbone, num_backbone_features=768, output_size=4, dropout_rate=0.3):
        super().__init__()
        self.backbone = backbone
        
        self.fc_layers = torch.nn.Sequential(
            torch.nn.LayerNorm(num_backbone_features), # Use LayerNorm for transformers
            torch.nn.Linear(num_backbone_features, num_backbone_features // 4),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(num_backbone_features // 4, output_size)
        )
    
    def forward(self, x):
        # The AST backbone expects input shape [batch_size, num_mels, time_frames]
        # It's different from vision models that expect [B, C, H, W]
        
        # The output is a special object from Hugging Face
        outputs = self.backbone(x)
        
        # We get the sequence of embeddings for all patches
        last_hidden_state = outputs.last_hidden_state 
        # last_hidden_state shape: [batch_size, sequence_length, embedding_dim]

        # To get a single feature vector for the whole audio clip,
        # we can average the embeddings across the sequence length dimension.
        mean_features = torch.mean(last_hidden_state, dim=1)
        
        # Now pass these features to your classifier head
        logits = self.fc_layers(mean_features)
        return logits


class RecommenderModelASTHybrid(torch.nn.Module):
    def __init__(self, backbone, num_spec_features=768, num_cekl_features=9, output_size=4, dropout_rate=0.3):
        super().__init__()
        self.backbone = backbone
        
        # The total number of features after concatenation
        total_features = num_spec_features + num_cekl_features

        # The classification head now operates on the combined feature vector
        self.fc_layers = torch.nn.Sequential(
            # Normalize the combined vector
            torch.nn.LayerNorm(total_features), 
            
            torch.nn.Linear(total_features, total_features // 4),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(total_features // 4, output_size)
        )
    
    def forward(self, x):
        # The dataloader will now provide x as a tuple or list
        spec_features, cekl_features = x
        
        # 1. Process the spectrogram features through the AST backbone
        outputs = self.backbone(spec_features)
        last_hidden_state = outputs.last_hidden_state
        mean_spec_features = torch.mean(last_hidden_state, dim=1)
        # mean_spec_features has shape [batch_size, 768]

        # cekl_features already has shape [batch_size, 9] from the dataloader
        
        # 2. Concatenate the two feature vectors along the feature dimension (dim=1)
        combined_features = torch.cat((mean_spec_features, cekl_features), dim=1)
        # combined_features has shape [batch_size, 768 + 9]
        
        # 3. Pass the hybrid vector to the classifier head
        logits = self.fc_layers(combined_features)
        return logits


def write_misclassified_labels_to_csv(misclassified_labels: dict, output_path):
    """
    Write misclassified labels to a CSV file.
    
    :param misclassified_labels: Dictionary with keys file_path, 
                                 'true_label', 'predicted_label'.
    """
    if not misclassified_labels:
        warnings.warn("No misclassified labels to write.")
        return
    
    df = pd.DataFrame(misclassified_labels)
    df.to_csv(output_path, index=False)
    print(f"Misclassified labels written to {output_path}")


class AudioDataset(Dataset):
    def __init__(self, audio_dataset_name, audio_dataset_dir, label_path, 
                 feature_type='mel_spec', target_sr=16000, 
                 n_mels=256, n_fft=400, hop_length=160): # Removed ce_kl_feature_dim
        self.audio_dataset_name = audio_dataset_name
        self.audio_dataset_dir = audio_dataset_dir
        self.label_path = label_path
        self.feature_type = feature_type
        self.target_sr = target_sr
        self.best_model_to_label_dict = {
            'All Failed': 0, 'CDiffuSE': 1, 'SGMSE': 2, 'StoRM': 3
        }

        self.audio_files = sorted([
            os.path.join(self.audio_dataset_dir, f) 
            for f in os.listdir(self.audio_dataset_dir) if f.endswith('.wav')
        ])
        if not self.audio_files:
            warnings.warn(f"No .wav files found in {self.audio_dataset_dir} for dataset {self.audio_dataset_name}")

        self.load_labels()

        if self.feature_type == 'mel_spec':
            self.mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.target_sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
            )
            self.amplitude_to_db_transform = torchaudio.transforms.AmplitudeToDB()
        elif self.feature_type == 'ce_kl':
            # Initialize user's NoiseDataPreprocessor
            # It takes audio_paths (directory string), sr, n_fft, hop_length.
            # The sr, n_fft, hop_length for its __init__ are less critical for our specific use of
            # extract_features_by_sample, as that method takes sr, and S_power is pre-computed.
            self.noise_data_preprocessor = NoiseDataPreprocessor(audio_paths=self.audio_dataset_dir)
            
            # Store n_fft and hop_length from args to compute STFT consistently
            self.n_fft = n_fft
            self.hop_length = hop_length
            # Power spectrogram, compatible with what NoiseDataPreprocessor.spectral_cross_entropy expects for S
            self.stft_transform = torchaudio.transforms.Spectrogram(
                n_fft=self.n_fft, hop_length=self.hop_length, power=2.0 # Power=2.0 for power spectrogram
            )
        else:
            raise ValueError(f"Unsupported feature_type: {self.feature_type}")

    def __len__(self):
        return len(self.audio_files)
    
    def load_labels(self):
        # (Same as before)
        if self.label_path is not None and os.path.exists(self.label_path):
            labels_df = pd.read_csv(self.label_path)
            breakpoint()
            if 'Unnamed: 0' in labels_df.columns:
                labels_df.rename(columns={'Unnamed: 0': 'filename'}, inplace=True)
            if 'filename' not in labels_df.columns or 'best_model' not in labels_df.columns:
                raise ValueError(f"CSV {self.label_path} must contain 'filename' and 'best_model' columns.")
            labels_df['best model numeric'] = labels_df['best_model'].map(self.best_model_to_label_dict)
            if labels_df['best model numeric'].isnull().any():
                unmapped = labels_df[labels_df['best model numeric'].isnull()]['best_model'].unique()
                warnings.warn(f"Unmapped labels in {self.label_path}: {unmapped}. These rows will be dropped.")
                labels_df.dropna(subset=['best model numeric'], inplace=True)
            labels_df.set_index('filename', inplace=True)
            self.labels_df = labels_df
        else:
            self.labels_df = None
            warnings.warn(f"Label path {self.label_path} not found or not provided for {self.audio_dataset_name}.")

    def __getitem__(self, idx):
        audio_file_path = self.audio_files[idx]
        audio_file_basename = os.path.basename(audio_file_path)

        if self.labels_df is None:
            raise RuntimeError(f"Labels not loaded for {self.audio_dataset_name}. Cannot get label for {audio_file_basename}.")
        try:
            label_val = self.labels_df.loc[audio_file_basename, 'best model numeric']
        except KeyError:
            raise ValueError(f"Label not found for '{audio_file_basename}' in {self.label_path}.")
        label = torch.tensor(int(label_val), dtype=torch.long)

        waveform, sample_rate = torchaudio.load(audio_file_path)
        if waveform.ndim > 1 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sample_rate != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sr)
            waveform = resampler(waveform)

        if self.feature_type == 'mel_spec':
            # if waveform is longer than timelength, pad it
            time_length = 15 # 15 seconds
            if waveform.shape[1] < self.target_sr * time_length:
                waveform = torch.nn.functional.pad(waveform, (0, self.target_sr * time_length - waveform.shape[1]))
            elif waveform.shape[1] > self.target_sr * time_length:
                waveform = waveform[:, :self.target_sr * time_length]

            features = self.mel_spectrogram_transform(waveform)
            features = self.amplitude_to_db_transform(features)
            if self.apply_spec_augment:
                # Option 1: Mask with zeros (if your model can handle it, or if you un-dB, mask, re-dB)
                # features = self.spec_augment_transforms(features) 

                # Option 2: Mask with the mean of the spectrogram (more common for dB scale)
                # Create masks
                freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=30) # Max 30 Mel bins
                time_mask = torchaudio.transforms.TimeMasking(time_mask_param=50)   # Max 50 time frames

                # Apply masking (can be done multiple times if desired by looping or using m_F/m_T params if available in a wrapper)
                features_masked = freq_mask(features.clone(), mask_value=features.mean()) # Use clone to avoid in-place modification if needed elsewhere
                features_masked = time_mask(features_masked, mask_value=features_masked.mean()) 
                features = features_masked
        elif self.feature_type == 'ce_kl':
            # 1. Compute Power Spectrogram using torchaudio
            # Output shape: (channel, freq_bins, time_frames), channel is 1 for mono
            s_power_torch = self.stft_transform(waveform)
            
            # Squeeze channel dim and convert to NumPy for NoiseDataPreprocessor
            s_power_numpy = s_power_torch.squeeze(0).numpy() # Shape: (freq_bins, time_frames)

            # 2. Pass the power spectrogram to user's feature extraction method
            # The `sr` argument is passed along as the user's method expects it.
            features_numpy = self.noise_data_preprocessor.extract_features_by_sample(s_power_numpy, self.target_sr)
            
            features = torch.tensor(features_numpy, dtype=torch.float32)
            if features.shape[0] != 9: # Should be 9 features
                 warnings.warn(f"CE/KL features for {audio_file_basename} has shape {features.shape}, expected (9,). Might cause errors.")
                 # Pad or truncate if necessary, though extract_features_by_sample should handle this.
                 if features.shape[0] < 9:
                     features = torch.cat((features, torch.zeros(9 - features.shape[0])))
                 else:
                     features = features[:9]
        else:
            raise ValueError(f"Unsupported feature_type in __getitem__: {self.feature_type}")
        
        return features, label
    

class PreSplitDataset(Dataset):
    def __init__(self, files, numeric_labels, feature_type, target_sr, 
                 n_mels, n_fft, hop_length, 
                 # SpecAugment specific parameters for clarity
                 apply_spec_augment=False, 
                 feature_extractor=None,
                 freq_mask_param=30, # Max width for frequency mask
                 time_mask_param=50,   # Max width for time mask
                 num_freq_masks=1,     # Number of frequency masks
                 num_time_masks=1,     # Number of time masks
                 # For NoiseDataPreprocessor initialization
                 base_audio_dataset_dir_for_ndp=None,
                 scaler=None,
                 use_hybrid_features=False,
                 cekl_feature_basedir=None): # Pass one of the original dirs for NDP init
        
        self.files = files
        self.numeric_labels = numeric_labels
        self.feature_type = feature_type
        self.target_sr = target_sr
        
        # Store augmentation parameters
        self.apply_spec_augment = apply_spec_augment
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.scaler = scaler # Optional scaler for CE/KL features
        self.feature_extractor = feature_extractor # Optional, for MelSpec pretraining
        self.use_hybrid_features = use_hybrid_features # For AST hybrid model
        self.cekl_features_base_dir = cekl_feature_basedir # Base dir for CE/KL features if hybrid model

        if self.feature_type == 'mel_spec':
            if self.feature_extractor is None:
                self.mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.target_sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
                self.amplitude_to_db_transform = torchaudio.transforms.AmplitudeToDB()
            
            if self.apply_spec_augment:
                # We will apply these manually in __getitem__ to allow for multiple masks
                # and masking with mean if desired.
                # For simplicity here, we'll use torchaudio's direct transforms.
                # If you need more control (e.g. mask_value=mean), do it manually.
                self.frequency_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=self.freq_mask_param)
                self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=self.time_mask_param)

        elif self.feature_type == 'ce_kl':
            if base_audio_dataset_dir_for_ndp is None:
                raise ValueError("base_audio_dataset_dir_for_ndp must be provided for ce_kl feature type in PreSplitDataset.")
            self.noise_data_preprocessor = NoiseDataPreprocessor(audio_paths=base_audio_dataset_dir_for_ndp, 
                                                                 sr=target_sr, n_fft=n_fft, hop_length=hop_length)
            self.stft_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2.0)
    
    def __len__(self):
        return len(self.files)
    
    def mix_max_scaling(self, features):
        """
        Apply Min-Max scaling to the features.
        :param features: Tensor of shape (n_features,).
        :return: Scaled features.
        """
        min_val = features.min()
        max_val = features.max()
        if max_val - min_val == 0:
            return features
        return (features - min_val) / (max_val - min_val)

    def __getitem__(self, idx):
        audio_file_path = self.files[idx]
        label = torch.tensor(self.numeric_labels[idx], dtype=torch.long)
        
        try:
            waveform, sample_rate = torchaudio.load(audio_file_path)
        except Exception as e:
            warnings.warn(f"Error loading audio file {audio_file_path}: {e}. Returning dummy data.")
            # Return dummy data of expected shapes to prevent crashing the batch collation
            # This is a fallback, ideally identify and fix/remove problematic files
            if self.feature_type == 'mel_spec':
                # Guessing n_mels and a short time dimension
                # Get n_mels from an initialized transform if possible, or use a default
                # For this example, let's assume n_mels was passed to init and stored, e.g. self.n_mels
                # Or get from mel_spectrogram_transform.n_mels if available.
                # For simplicity, hardcoding a common value here for the dummy.
                dummy_features = torch.zeros((1, 128, 100), dtype=torch.float) 
            elif self.feature_type == 'ce_kl':
                dummy_features = torch.zeros(9, dtype=torch.float)
            return dummy_features, torch.tensor(0, dtype=torch.long) # Dummy label

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
            
        if waveform.shape[0] > 1: # If stereo, convert to mono
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed to match the extractor's required sample rate
        if self.feature_extractor is not None:
            target_sr = self.feature_extractor.sampling_rate
        else:
            target_sr = self.target_sr # Fallback for non-AST models

        if sample_rate != target_sr:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(waveform)

        if self.feature_type == 'mel_spec':
            if self.feature_extractor is not None: # AST model path
                # The extractor expects a 1D waveform or a batch of 1D waveforms
                # Squeeze the channel dimension before passing it in.
                waveform_squeezed = waveform.squeeze(0)
                
                inputs = self.feature_extractor(
                    waveform_squeezed, 
                    sampling_rate=self.feature_extractor.sampling_rate, 
                    return_tensors="pt"
                )
                
                # The output 'input_values' has shape [1, num_mels, time_frames]
                # Squeeze the batch dimension that the extractor adds.
                features = inputs['input_values'].squeeze(0)
                # 'features' now has the correct shape [num_mels, time_frames]

            else: # Original timm model path
                # Your padding and old feature extraction logic
                time_length = 15
                if waveform.shape[1] < self.target_sr * time_length:
                    waveform = torch.nn.functional.pad(waveform, (0, self.target_sr * time_length - waveform.shape[1]))
                elif waveform.shape[1] > self.target_sr * time_length:
                    waveform = waveform[:, :self.target_sr * time_length]

                features = self.mel_spectrogram_transform(waveform)
                features = self.amplitude_to_db_transform(features)
                features = self.mix_max_scaling(features)
                
                feature_order1 = torchaudio.transforms.ComputeDeltas()(features)
                feature_order1 = self.mix_max_scaling(feature_order1)
                feature_order2 = torchaudio.transforms.ComputeDeltas()(feature_order1)
                feature_order2 = self.mix_max_scaling(feature_order2)
                
                # Here you need to stack the channels correctly for timm models
                # The original features already have a channel dim, so just cat them
                # Timm expects 3 channels
                features = torch.cat((features, feature_order1, feature_order2), dim=0)
            if self.apply_spec_augment:
                for _ in range(self.num_freq_masks):
                    features = self.frequency_masking(features) # Masks with 0 by default
                for _ in range(self.num_time_masks):
                    features = self.time_masking(features)   # Masks with 0 by default
            if self.use_hybrid_features:
                relative_path = Path(audio_file_path).relative_to('/success_fail_estimation/') # Adjust this base path
                cekl_feature_path = Path(self.cekl_features_base_dir) / relative_path.with_suffix('.npy')

                # 2. Load the features from the .npy file
                cekl_features_numpy = np.load(cekl_feature_path)
                cekl_features = torch.tensor(cekl_features_numpy, dtype=torch.float32)
                return (features, cekl_features), label  # Return both MelSpec and CE/KL features for hybrid model
        elif self.feature_type == 'ce_kl':
            s_power_torch = self.stft_transform(waveform)
            s_power_numpy = s_power_torch.squeeze(0).numpy()
            features_numpy = self.noise_data_preprocessor.extract_features_by_sample(s_power_numpy, self.target_sr)
            if self.scaler:
                features_numpy = self.scaler.transform(features_numpy.reshape(1, -1))
                features_numpy = features_numpy.flatten() # Reshape to 2D for scaler
            features = torch.tensor(features_numpy, dtype=torch.float32)

            if features.shape[0] != 9: # Ensure correct shape
                features = (torch.cat((features, torch.zeros(9 - features.shape[0]))) if features.shape[0] < 9 else features[:9])
        else:
            raise ValueError(f"Unsupported feature_type in PreSplitDataset __getitem__: {self.feature_type}")
            
        return features, label


class ModelRecommenderPipeline:
    def __init__(self, model, num_epochs=10, base_learning_rate=0.001, device=None, class_weights=None,
                 optimizer_grouped_params = None, warmup_epochs = 0,base_weight_decay=1e-3): # Added class_weights
        self.model = model
        self.num_epochs = num_epochs
        self.base_learning_rate = base_learning_rate
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_weights = class_weights # Store class weights
        self.warmup_epochs = warmup_epochs # Store warmup epochs
        self.optimizer_grouped_params = optimizer_grouped_params # Store optimizer params if provided
        self.current_epochs = 0
        self.model.to(self.device)
        # Initialize criterion with class weights if provided
        self.criterion = torch.nn.CrossEntropyLoss(weight=None,label_smoothing=0.1) # Default to no class weights, label smoothing 0.1

        if optimizer_grouped_params:
            self.optimizer = torch.optim.AdamW(optimizer_grouped_params, 
                                               lr=base_learning_rate, # Acts as default if a group has no lr
                                               weight_decay=base_weight_decay) 
            # Store initial LRs for warmup scaling if groups are used
            self.initial_group_lrs = [pg.get('lr', base_learning_rate) for pg in optimizer_grouped_params]
            pass
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.base_learning_rate, weight_decay=0.001)
        # Example: ReduceLROnPlateau scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=2, factor=0.5, verbose=False)

    def _apply_warmup_lr(self):
        if self.current_epochs < self.warmup_epochs:
            warmup_start_factor = 0.01 
            # Ensure self.warmup_epochs is > 0 to avoid division by zero if current_epoch is 0
            # and warmup_epochs is 0 or 1.
            effective_warmup_epochs = max(1, self.warmup_epochs) # Avoid div by zero if warmup_epochs is 0
            
            # Linear scaling from warmup_start_factor to 1.0 over warmup_epochs
            # Example: epoch 0 -> scale = 0/2 = 0; epoch 1 -> scale = 1/2 = 0.5; epoch 2 -> scale = 2/2 = 1 (if warmup_epochs=2)
            # Corrected scaling:
            if self.current_epochs == 0 and effective_warmup_epochs > 0 : # First epoch of warmup
                current_scale = warmup_start_factor
            elif effective_warmup_epochs > 0 :
                 current_scale = warmup_start_factor + (1.0 - warmup_start_factor) * (self.current_epochs / (effective_warmup_epochs -1 + 1e-8) ) # +1e-8 to avoid div by zero if warmup_epochs is 1
            else: # No warmup
                current_scale = 1.0
            current_scale = min(current_scale, 1.0) # Cap at 1.0

            for i, param_group in enumerate(self.optimizer.param_groups):
                # Scale the group's initial/target LR
                param_group['lr'] = self.initial_group_lrs[i] * current_scale 
            
            if self.current_epochs == 0: # Print only on first warmup epoch for brevity
                 print(f"Warmup Epoch {self.current_epochs+1}/{self.warmup_epochs}. LRs scaled by ~{current_scale:.2f}. Group 0 LR: {self.optimizer.param_groups[0]['lr']:.2e}")

        elif self.current_epochs == self.warmup_epochs: # First epoch *after* warmup
            # Ensure target LRs are fully restored for all groups
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.initial_group_lrs[i]
            print(f"Warmup complete. LRs restored. Group 0 LR: {self.optimizer.param_groups[0]['lr']:.2e}")

    def train_one_epoch(self, train_loader, scaler=None):
        self._apply_warmup_lr()
        self.model.train()
        running_loss = 0.0
        for batch_idx, (features, labels) in enumerate(train_loader):
            if isinstance(features, (list, tuple)):
                # Move each tensor in the tuple to the device
                features = [f.to(self.device) for f in features]
            else:
                # Standard case for single-feature models
                features = features.to(self.device)
            labels = labels.to(self.device)           
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels) # Criterion now uses weights
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        self.current_epochs += 1
        return avg_loss

    def evaluate(self, data_loader, calculate_val_loss=False, scaler=None): # Added flag to calculate val_loss for scheduler
        self.model.eval()
        all_predictions, all_labels = [], []
        running_val_loss = 0.0
        with torch.no_grad():
            for features, labels in data_loader:
                if scaler:
                    features = scaler.transform(features.numpy())
                    features = torch.tensor(features, dtype=torch.float32)
                if isinstance(features, (list, tuple)):
                    # Move each tensor in the tuple to the device
                    features = [f.to(self.device) for f in features]
                else:
                    # Standard case for single-feature models
                    features = features.to(self.device)

                labels = labels.to(self.device)                
                outputs = self.model(features)
                if calculate_val_loss:
                    loss = self.criterion(outputs, labels) # Use the same criterion (can be unweighted for val if desired)
                    running_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = running_val_loss / len(data_loader) if calculate_val_loss and len(data_loader) > 0 else None

        if not all_labels: return 0.0, 0.0, 0.0, 0.0, [], avg_val_loss

        cls_report = classification_report(all_labels, all_predictions, output_dict=True)
        accuracy = cls_report['accuracy']
        f1 = cls_report['macro avg']['f1-score']
        precision = cls_report['macro avg']['precision']
        recall = cls_report['macro avg']['recall']
        
        return accuracy, f1, precision, recall, all_predictions, avg_val_loss # Return val_loss
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    # In your recommender_pipeline class

    def analyze_gatekeeper_performance(self, true_labels, gatekeeper_predictions):
        """Analyzes the binary performance of the Gatekeeper on the full validation set."""
        # Convert 4-class true labels to binary for Gatekeeper evaluation
        binary_true_labels = [1 if label == 2 else 0 for label in true_labels]
        
        print("\n--- Gatekeeper Standalone Performance Analysis ---")
        print(classification_report(binary_true_labels, gatekeeper_predictions, target_names=['NOT_SGMSE', 'IS_SGMSE'], zero_division=0))
        cm = confusion_matrix(binary_true_labels, gatekeeper_predictions)
        print("Gatekeeper Confusion Matrix:\n", cm)

    def analyze_correctly_classified(self, gatekeeper_conf, expert_conf):
        """Analyzes the confidence scores of correctly classified samples."""
        print("\n--- Correctly Classified Samples Confidence ---")

        if gatekeeper_conf:
            # Use np.mean and np.std for robustness
            mean_gk_conf, std_gk_conf = np.mean(gatekeeper_conf), np.std(gatekeeper_conf)
            print(f"Gatekeeper Correct Confidence ({len(gatekeeper_conf)} samples):")
            print(f"  - Average: {mean_gk_conf:.4f} (Std: {std_gk_conf:.4f})")
        else:
            print("Gatekeeper made no correct 'IS_SGMSE' classifications.")

        if expert_conf:
            mean_ex_conf, std_ex_conf = np.mean(expert_conf), np.std(expert_conf)
            print(f"Expert Correct Confidence ({len(expert_conf)} samples on routed data):")
            print(f"  - Average: {mean_ex_conf:.4f} (Std: {std_ex_conf:.4f})")
        else:
            print("Expert made no correct classifications on the data it received.")

    def analyze_error_contribution(self, gatekeeper_fp_count, gatekeeper_fn_count, expert_error_count, total_errors):
        """Calculates and prints the source of all pipeline errors."""
        print("\n--- Pipeline Error Contribution Analysis ---")
        if total_errors == 0:
            print("No errors found in the pipeline. Perfect performance!")
            return
            
        print(f"Total Errors: {total_errors}")
        
        gatekeeper_fp_percent = (gatekeeper_fp_count / total_errors) * 100
        gatekeeper_fn_percent = (gatekeeper_fn_count / total_errors) * 100
        expert_error_percent = (expert_error_count / total_errors) * 100
        
        print(f"- {gatekeeper_fp_percent:.1f}% ({gatekeeper_fp_count}) from Gatekeeper False Positives (Incorrectly sent to SGMSE).")
        print(f"- {gatekeeper_fn_percent:.1f}% ({gatekeeper_fn_count}) from Gatekeeper False Negatives (Incorrectly sent to Expert).")
        print(f"- {expert_error_percent:.1f}% ({expert_error_count}) from Expert Errors (on correctly routed data).")
            


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Recommender Pipeline")
    parser.add_argument('--feature_type', type=str, required=True, choices=['mel_spec', 'ce_kl'])
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--warmup_epochs', type=int, default=0, help="Number of warmup epochs for learning rate scaling")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="Base weight decay for AdamW optimizer")
    # STFT parameters (used for MelSpec and for STFT before CE/KL)
    parser.add_argument('--n_mels', type=int, default=128, help='Number of Mel bands for Mel spectrogram')
    parser.add_argument('--n_fft', type=int, default=400, help='N_FFT for STFT (MelSpectrogram and CE/KL)')
    parser.add_argument('--hop_length', type=int, default=160, help='Hop length for STFT (MelSpectrogram and CE/KL)')
    parser.add_argument('--num_workers', type=int, default=0) # Default 0 for easier debugging
    parser.add_argument('--create_dummy_data', action='store_true', default=False)
    parser.add_argument('--use_class_weights', action='store_true', help="Use class weighting")
    parser.add_argument('--apply_spec_augment', action='store_true', default=False,
                        help="Apply SpecAugment to MelSpectrogram features")
    parser.add_argument('--freq_mask_param', type=int, default=27, help="SpecAugment: Max freq mask width") # Common value for 128 mels
    parser.add_argument('--time_mask_param', type=int, default=70, help="SpecAugment: Max time mask width") # Common value
    parser.add_argument('--num_freq_masks', type=int, default=1, help="SpecAugment: Number of freq masks")
    parser.add_argument('--num_time_masks', type=int, default=1, help="SpecAugment: Number of time masks")
    parser.add_argument('--use_weighted_sampler', action='store_true', default=True,
                        help="Use WeightedRandomSampler for training data to handle class imbalance")
    parser.add_argument('--pretrained_models', type=str, default=None,help="architecture of pretrained models to use",
                        choices=['resnet18','efficientnet_b2','ast_finetuned_audioset'])
    parser.add_argument('--training_stage', type=str, default='full_4_class', 
                        choices=['full_4_class', 'gatekeeper', 'expert', 'evaluate_pipeline'],
                        help="Specify the training or evaluation stage.")
    parser.add_argument('--gatekeeper_model_path', type=str, default="./gatekeeper_model_fold_{fold}_best_hybrid.pth",
                        help="Path template to load/save the gatekeeper model.")
    parser.add_argument('--expert_model_path', type=str, default="./expert_model_fold_{fold}_best_hybrid.pth",
                        help="Path template to load/save the expert model.")
    parser.add_argument('--use_hybrid_features', action='store_true', default=False,)
    parser.add_argument('--ce_kl_features_basedir', type=str, default=None,
                        help="Base directory for CE/KL features if using hybrid model.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Setup (Update paths as needed) ---
    audio_dataset_names = [
        # 'dns', 
        #'voicebank', 
        'crema_mssnsd', 
        #'voicebank_train',
        #'ljspeech_esc50'
    ]
    base_data_dir = '/success_fail_estimation/' 
    audio_dataset_dirs = [
    #    os.path.join(base_data_dir, 'noisy_blind_testset_v3_challenge_withSNR_16k'),
    #    os.path.join(base_data_dir, 'voicebank/noisy_testset_wav'),
        os.path.join(base_data_dir, 'crema_mssnsd_non_stationary/test/noisy'),
    #    os.path.join(base_data_dir, 'sgmse/noisy_trainset_28spk_wav'),
    #    os.path.join(base_data_dir, 'ljspeech_esc50/noisy')
    ]
    label_paths = [
    #    os.path.join(base_data_dir, 'best_model_dnsmos_records_dns.csv'),
    #    os.path.join(base_data_dir, 'best_model_dnsmos_records_voicebank.csv'),
        os.path.join(base_data_dir, 'best_model_dnsmos_records_crema_mssnsd.csv'),
    #    os.path.join(base_data_dir, 'best_model_dnsmos_records_voicebank_train.csv'),
    #    os.path.join(base_data_dir, 'best_model_dnsmos_records_ljspeech_esc50.csv')
    ]
    hf_model_path = 'MIT/ast-finetuned-audioset-10-10-0.4593'

    cekl_features_dict = {
        'diagonal variance of CE':None,
        'mean off-diagonal of CE':None,
        'variance off-diagonal of CE':None,
        'mean adjacent frame of CE':None,
        'variance adjacent frame of CE':None,
        'mean off-diagonal of KL divergence':None,
        'variance off-diagonal of KL divergence':None,
        'mean adjacent frame of KL divergence':None,
        'variance adjacent frame of KL divergence':None
    }

    if args.create_dummy_data:
        # (Dummy data creation logic - same as before, ensure it creates .wav files)
        print("Attempting to create dummy data...")
        dummy_audio_files_per_dataset = 5 
        dummy_labels_map_inv = {0: 'All Failed', 1: 'CDiffuSE', 2: 'SGMSE', 3: 'StoRM'}
        for i, data_dir in enumerate(audio_dataset_dirs):
            os.makedirs(data_dir, exist_ok=True)
            label_file = label_paths[i]
            print(f"Creating dummy files for: {data_dir}")
            dummy_filenames, temp_labels = [], []
            for j in range(dummy_audio_files_per_dataset):
                filename = f"dummy_audio_{audio_dataset_names[i]}_{j}.wav"
                filepath = os.path.join(data_dir, filename)
                dummy_filenames.append(filename)
                temp_labels.append(dummy_labels_map_inv[np.random.randint(0,4)])
                if not os.path.exists(filepath):
                    sample_rate = 16000 # target_sr for dummy data
                    dummy_waveform = torch.sin(2 * np.pi * torch.randint(220, 880, (1,)).item() * torch.arange(0, 0.5, 1/sample_rate).float()).unsqueeze(0)
                    torchaudio.save(filepath, dummy_waveform, sample_rate)
            df_data = {'filename': dummy_filenames, 'best model': temp_labels} # Use 'filename'
            dummy_df = pd.DataFrame(df_data)
            os.makedirs(os.path.dirname(label_file), exist_ok=True)
            dummy_df.to_csv(label_file, index=False)
            print(f"Created dummy label file: {label_file} with {len(dummy_df)} entries.")
    # --- End of Dummy Data Creation ---
    breakpoint()
    datasets_list = []
    all_files_global = []
    all_labels_global_numeric = []
    # This mapping is needed for PreSplitDataset if feature_type is ce_kl
    # It's a bit of a hack due to NoiseDataPreprocessor's __init__ needing a dir.
    # We'll use the *original* dataset directory associated with each file.
    all_original_dirs_global = [] 

    # Define the label mapping once, ensure consistency
    # This should match what's in AudioDataset if you were using it.
    global_best_model_to_label_dict = {'All Failed': 0, 'CDiffuSE': 1, 'SGMSE': 2, 'StoRM': 3}

    expert_label_map = {0: 0, 1: 1, 3: 2} # original label -> new 0,1,2 label

    for i in range(len(audio_dataset_names)):
        dataset_name = audio_dataset_names[i]
        audio_dataset_dir = audio_dataset_dirs[i]
        label_path = label_paths[i]
        if not os.path.isdir(audio_dataset_dir):
            warnings.warn(f"Directory {audio_dataset_dir} not found. Skipping dataset {dataset_name}.")
            continue
        
        current_audio_files = sorted([
            os.path.join(audio_dataset_dir, f) 
            for f in os.listdir(audio_dataset_dir) if f.endswith('.wav')
        ])
        if not current_audio_files:
            warnings.warn(f"No .wav files found in {audio_dataset_dir} for dataset {dataset_name}")
            continue

        df_labels_current = None
        if label_path is not None and os.path.exists(label_path):
            df_labels_current = pd.read_csv(label_path)
            if 'Unnamed: 0' in df_labels_current.columns:
                df_labels_current.rename(columns={'Unnamed: 0': 'filename'}, inplace=True)
            if 'filename' not in df_labels_current.columns or 'best_model' not in df_labels_current.columns:
                warnings.warn(f"Skipping {label_path}, missing 'filename' or 'best model' columns.")
                df_labels_current = None
            else:
                df_labels_current['best model numeric'] = df_labels_current['best_model'].map(global_best_model_to_label_dict)
                df_labels_current.dropna(subset=['best model numeric'], inplace=True) # Drop rows with unmapped/NaN labels
                df_labels_current.set_index('filename', inplace=True)
        
        if df_labels_current is None:
            warnings.warn(f"No labels loaded for {dataset_name} from {label_path}. Skipping this dataset for global list.")
            continue

        for audio_file_path in current_audio_files:
            audio_file_basename = os.path.basename(audio_file_path)
            if audio_file_basename in df_labels_current.index:
                original_label = df_labels_current.loc[audio_file_basename, 'best model numeric']
                if args.training_stage == 'gatekeeper':
                    # Binary labels: 1 if SGMSE, 0 otherwise
                    label_val = 1 if original_label == 2 else 0
                    all_files_global.append(audio_file_path)
                    all_labels_global_numeric.append(label_val)
                    all_original_dirs_global.append(audio_dataset_dir)

                elif args.training_stage == 'expert':
                    # Only include non-SGMSE samples
                    if original_label != 2:
                        label_val = expert_label_map[original_label] # Re-map 0,1,3 to 0,1,2
                        all_files_global.append(audio_file_path)
                        all_labels_global_numeric.append(label_val)
                        all_original_dirs_global.append(audio_dataset_dir)

                elif args.training_stage in ['full_4_class', 'evaluate_pipeline']:
                    # For standard training or final evaluation, use original 4-class labels
                    label_val = int(original_label)
                    all_files_global.append(audio_file_path)
                    all_labels_global_numeric.append(label_val)
                    all_original_dirs_global.append(audio_dataset_dir)
            # else:
                # warnings.warn(f"Label not found for {audio_file_basename} in {label_path}. Skipping file.")

    if not all_files_global:
        raise ValueError("No audio files with corresponding labels were found across all datasets. Check paths and label CSVs.")
    
    print(f"Total usable samples collected across all datasets: {len(all_files_global)}")

    # --- K-Fold Cross-Validation ---
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    # KFold splits indices from 0 to len(all_files_global)-1
    for fold, (train_idx_global, val_idx_global) in enumerate(k_fold.split(all_files_global)):
        print(f"\n--- Fold {fold+1}/5 ---")

        # Get file paths, labels, and original directories for this fold's train and val sets
        train_files_fold = [all_files_global[i] for i in train_idx_global]
        train_labels_fold_numeric = [all_labels_global_numeric[i] for i in train_idx_global]
        # For CE/KL, NDP needs a base path for init. We use the specific original path for each file.
        # However, PreSplitDataset takes only one base_audio_dataset_dir_for_ndp.
        # This means for CE/KL, all files in a fold's PreSplitDataset will use the *same* base dir for NDP __init__.
        # This is okay IF NDP's extract_features_by_sample doesn't rely on its self.audio_paths for per-sample processing.
        # We'll use the directory of the FIRST training file as the representative for NDP init.
        # This is a simplification; a more robust NDP wouldn't need this.
        representative_dir_for_ndp = all_original_dirs_global[train_idx_global[0]] if len(train_idx_global) > 0 else (all_original_dirs_global[val_idx_global[0]] if len(val_idx_global) > 0 else base_data_dir)


        val_files_fold = [all_files_global[i] for i in val_idx_global]
        val_labels_fold_numeric = [all_labels_global_numeric[i] for i in val_idx_global]

        feature_extractor = None
        # --- Model Initialization ---
        if args.feature_type == 'mel_spec':
            if not args.pretrained_models:
                # Default to custom model if no pretrained model specified
                model = RecommenderModelMelSpec(hidden_size=args.hidden_size, dropout_rate=args.dropout_rate)
            else:
                if args.pretrained_models != 'ast_finetuned_audioset':
                    backbone = timm.create_model(
                        model_name=args.pretrained_models, 
                        pretrained=True, 
                        num_classes=0,
                        in_chans=3
                    )
                    if backbone is not None: # Check if model creation was successful
                        for name, param in backbone.named_parameters(): # Freeze all parameters first
                            # layer_name = name.split('.')[0]
                            # sub_layer_name = name.split('.')[1] if len(name.split('.')) > 1 else ''
                            # if layer_name == 'classifier' or layer_name == 'fc': # Adjust based on model architecture
                            #     param.requires_grad = True
                            # elif layer_name == 'conv1' or layer_name == 'bn1' or layer_name=='conv_stem': # For ResNets, 'fc' is the final layer
                            #     param.requires_grad = True
                            # elif layer_name == 'layer4' or layer_name == 'conv_head' or layer_name=='bn2': # For ResNets, unfreeze last block
                            #     param.requires_grad = True
                            # elif layer_name == 'blocks' and sub_layer_name == '6':
                            #     param.requires_grad = True
                            # else:
                            param.requires_grad = True
                    model = RecommenderModelMelSpecPretrain(
                        backbone=backbone,
                        dropout_rate=args.dropout_rate, 
                        num_backbone_features=512 if args.pretrained_models.startswith('resnet') else 1408)
                else:
                    feature_extractor = ASTFeatureExtractor.from_pretrained(hf_model_path)
                    backbone = ASTModel.from_pretrained(hf_model_path)
                    if args.training_stage == 'gatekeeper':
                        output_size = 2
                        print("Initializing AST model for GATEKEEPER training (2 classes).")
                    elif args.training_stage == 'expert':
                        output_size = 3
                        print("Initializing AST model for EXPERT training (3 classes).")
                    else: # full_4_class or evaluate_pipeline
                        output_size = 4
                        print("Initializing AST model for FULL training (4 classes).")
                    if args.use_hybrid_features:
                        model = RecommenderModelASTHybrid(
                            backbone=backbone,
                            output_size=output_size, # Use the dynamic output size
                            dropout_rate=args.dropout_rate,
                        )
                    else:
                        model = RecommenderModelAST(
                            backbone=backbone,
                            output_size=output_size, # Use the dynamic output size
                            dropout_rate=args.dropout_rate,
                        )
                lr_head = args.learning_rate  # e.g., 5e-5 (Your current args.learning_rate)
                lr_backbone = args.learning_rate / 2 # e.g., 1e-5 (Typically smaller for backbone)
                                                        # Make this an arg if you want more control: args.backbone_lr

                optimizer_grouped_parameters = [
                    {'params': model.fc_layers.parameters(), 'lr': lr_head, 'weight_decay': args.weight_decay}, # Add separate WD if desired
                    {'params': model.backbone.parameters(), 'lr': lr_backbone, 'weight_decay': args.weight_decay}
                ]
        elif args.feature_type == 'ce_kl':
            model = RecommenderModelCeKl(hidden_size=args.hidden_size, dropout_rate=args.dropout_rate)
        # --- End Model Initialization ---

        train_dataset_fold = PreSplitDataset(
            files=train_files_fold, numeric_labels=train_labels_fold_numeric, 
            feature_type=args.feature_type, target_sr=16000,
            n_mels=args.n_mels, n_fft=args.n_fft, hop_length=args.hop_length,
            apply_spec_augment=(args.apply_spec_augment if args.feature_type == 'mel_spec' else False), # Only for mel_spec
            freq_mask_param=args.freq_mask_param, time_mask_param=args.time_mask_param,
            num_freq_masks=args.num_freq_masks, num_time_masks=args.num_time_masks,
            base_audio_dataset_dir_for_ndp=representative_dir_for_ndp,
            feature_extractor=feature_extractor if feature_extractor else None, # Only for AST pretraining
            use_hybrid_features=args.use_hybrid_features, # Only for AST hybrid model
            cekl_feature_basedir= args.ce_kl_features_basedir if args.use_hybrid_features else None
        )
        val_dataset_fold = PreSplitDataset(
            files=val_files_fold, numeric_labels=val_labels_fold_numeric,
            feature_type=args.feature_type, target_sr=16000,
            n_mels=args.n_mels, n_fft=args.n_fft, hop_length=args.hop_length,
            apply_spec_augment=False, # Never augment validation set
            base_audio_dataset_dir_for_ndp=representative_dir_for_ndp, # Use same representative dir
            feature_extractor=feature_extractor if feature_extractor else None, # Only for AST pretraining
            use_hybrid_features=args.use_hybrid_features, # Only for AST hybrid model
            cekl_feature_basedir= args.ce_kl_features_basedir if args.use_hybrid_features else None
        )
        if args.feature_type == 'ce_kl':
            scaler = prepare_feature_standard_scaler_by_fold(train_dataset_fold)
        else:
            scaler = None

        train_dataset_fold.scaler = scaler # Attach scaler to dataset if needed
        val_dataset_fold.scaler = scaler # Attach scaler to validation dataset if needed

        print(f"Fold {fold+1}: Train dataset size: {len(train_dataset_fold)}, Val dataset size: {len(val_dataset_fold)}")
        if len(train_dataset_fold) == 0 or len(val_dataset_fold) == 0:
            warnings.warn(f"Fold {fold+1} has an empty train or validation set. Skipping fold.")
            continue

        # --- WeightedRandomSampler for Training Data ---
        sampler_for_train = None
        if args.use_weighted_sampler and len(train_labels_fold_numeric) > 0:
            print(f"Fold {fold+1} Training Class Distribution for Sampler: {Counter(train_labels_fold_numeric)}")
            label_counts_fold = Counter(train_labels_fold_numeric)
            
            per_sample_weights = []
            for label_val in train_labels_fold_numeric:
                class_count = label_counts_fold.get(label_val, 1) # Default to 1 to avoid div by zero
                weight = 1.0 / max(1, class_count) # Ensure count is at least 1
                per_sample_weights.append(weight)
            
            sampler_for_train = WeightedRandomSampler(
                weights=torch.DoubleTensor(per_sample_weights),
                num_samples=len(train_labels_fold_numeric),
                replacement=True
            )
            print(f"Fold {fold+1}: Using WeightedRandomSampler for training.")
        # --- End Sampler ---



        # --- Pipeline (Loss is unweighted as sampler handles balancing) ---
        # recommender_pipeline = ModelRecommenderPipeline(
        #     model=model, num_epochs=args.num_epochs, learning_rate=args.learning_rate, 
        #     device=device, class_weights=None # Crucially, class_weights=None if sampler is used
        # )

        recommender_pipeline = ModelRecommenderPipeline(
            model=model,
            num_epochs=args.num_epochs,
            base_learning_rate=args.learning_rate, # Base LR for reference or non-grouped case
            device=device,
            class_weights=None, # Should be None if using sampler
            optimizer_grouped_params=optimizer_grouped_parameters if optimizer_grouped_parameters else None, # Pass grouped params if using pretrained model
            warmup_epochs=args.warmup_epochs,
            base_weight_decay=args.weight_decay # Pass a base weight decay
        )
        # --- End Pipeline ---
        
        # --- DataLoaders ---
        train_shuffle = sampler_for_train is None # Shuffle if not using sampler
        train_loader = DataLoader(train_dataset_fold, batch_size=args.batch_size, 
                                  sampler=sampler_for_train, shuffle=train_shuffle, # Pass sampler, set shuffle accordingly
                                  num_workers=args.num_workers, pin_memory=(device.type == 'cuda'),drop_last=True)
        val_loader = DataLoader(val_dataset_fold, batch_size=args.batch_size, shuffle=False, 
                                num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
        # --- End DataLoaders ---
        
        # --- Training Loop for Fold ---
        if args.training_stage in ['gatekeeper', 'expert', 'full_4_class']:
            print(f"Training stage: {args.training_stage}. Proceeding with training.")
            best_val_f1_fold = 0.0
            epochs_no_improve = 0
            patience_early_stopping = 10 # Increased patience

            for epoch in range(recommender_pipeline.num_epochs):
                avg_train_loss = recommender_pipeline.train_one_epoch(train_loader)
                val_acc, val_f1, val_prec, val_rec, _, avg_val_loss_epoch = recommender_pipeline.evaluate(val_loader, calculate_val_loss=True)

                current_lr = recommender_pipeline.optimizer.param_groups[0]["lr"]
                if avg_val_loss_epoch is not None:
                    recommender_pipeline.scheduler.step(avg_val_loss_epoch)
                
                print(f'Fo {fold+1}, Ep [{epoch+1}/{args.num_epochs}], LR: {current_lr:.1e}, TrLoss: {avg_train_loss:.4f} | '
                    f'VaLoss: {avg_val_loss_epoch if avg_val_loss_epoch is not None else "N/A":.4f}, VaAcc: {val_acc:.4f}, VaF1: {val_f1:.4f}, VaPrec: {val_prec:.4f}, VaRec: {val_rec:.4f}')
                
                if val_f1 > best_val_f1_fold:
                    best_val_f1_fold = val_f1
                    epochs_no_improve = 0
                    # torch.save(model.state_dict(), f"model_{args.feature_type}_{args.pretrained_models}_fold{fold+1}_best_f1.pth") # Optional: save best model
                    if args.training_stage == 'gatekeeper':
                        save_path = args.gatekeeper_model_path.format(fold=fold + 1)
                    elif args.training_stage == 'expert':
                        save_path = args.expert_model_path.format(fold=fold + 1)
                    else: # full_4_class
                        save_path = f"model_full_4_class_fold{fold+1}_best_f1.pth"
                    
                    print(f"New best model found! Saving to {save_path}")
                    torch.save(model.state_dict(), save_path)
                else:
                    epochs_no_improve += 1
                
                if epochs_no_improve >= patience_early_stopping:
                    print(f"Early stopping triggered at epoch {epoch+1} for fold {fold+1} due to no F1 improvement.")
                    break
            # --- End Training Loop ---
        
        elif args.training_stage == 'evaluate_pipeline':
            print("\n--- Evaluating Two-Stage Pipeline for Fold {} ---".format(fold + 1))

            # 1. Load models (your existing code is fine here)
            gatekeeper_model = RecommenderModelASTHybrid(backbone=ASTModel.from_pretrained(hf_model_path), output_size=2)
            gatekeeper_path = args.gatekeeper_model_path.format(fold=fold + 1)
            gatekeeper_model.load_state_dict(torch.load(gatekeeper_path))
            gatekeeper_model.to(device)
            gatekeeper_model.eval()

            expert_model = RecommenderModelASTHybrid(backbone=ASTModel.from_pretrained(hf_model_path), output_size=3)
            expert_path = args.expert_model_path.format(fold=fold + 1)
            expert_model.load_state_dict(torch.load(expert_path))
            expert_model.to(device)
            expert_model.eval()

            val_loader = DataLoader(val_dataset_fold, batch_size=args.batch_size, shuffle=False)

            # --- Refactored Data Collection ---
            all_true_labels = []
            all_final_preds = []
            all_gatekeeper_preds = [] # For Gatekeeper-specific analysis

            # Data for error analysis
            gatekeeper_false_positives = [] # True is NOT_SGMSE (0,1,3), Pred is SGMSE (2)
            gatekeeper_false_negatives = [] # True is SGMSE (2), Pred is NOT_SGMSE
            expert_errors = []              # Gatekeeper was correct, but Expert was wrong

            # Data for confidence analysis of correct predictions
            gatekeeper_correct_confidence = []
            expert_correct_confidence = []

            # Data to analyze expert performance on all data it was given
            # We will store tuples of (true_label, expert_predicted_label)
            expert_results_on_routed_data = []

            # Data for "hardest samples" analysis
            low_confidence_samples = []
            expert_correct_samples = []

            expert_prediction_map_inv = {0: 0, 1: 1, 2: 3} # from 0,1,2 -> original 0,1,3
            # A map for the expert's own 3-class evaluation
            expert_label_map = {0: 0, 1: 1, 3: 2} # original 0,1,3 -> expert's 0,1,2

            with torch.no_grad():
                for batch_idx, ((spec_features, cekl_features), labels) in enumerate(val_loader):
                    spec_features = spec_features.to(device)
                    cekl_features = cekl_features.to(device) if cekl_features is not None else None
                    batch_true_labels = labels.cpu().numpy()

                    # --- PREDICTION LOGIC ---
                    gatekeeper_logits = gatekeeper_model((spec_features,cekl_features))
                    gatekeeper_probs = torch.softmax(gatekeeper_logits, dim=1)
                    gatekeeper_confidences, gatekeeper_preds = torch.max(gatekeeper_probs, 1)

                    expert_logits = expert_model((spec_features,cekl_features))
                    expert_probs = torch.softmax(expert_logits, dim=1)
                    expert_confidences, expert_preds_raw = torch.max(expert_probs, 1)

                    # Store all predictions for later analysis
                    all_true_labels.extend(batch_true_labels)
                    all_gatekeeper_preds.extend(gatekeeper_preds.cpu().numpy())

                    # --- PROCESS BATCH RESULTS ---
                    for i in range(len(batch_true_labels)):
                        true_label = batch_true_labels[i]
                        gk_pred = gatekeeper_preds[i].item()
                        gk_confidence = gatekeeper_confidences[i].item()
                        
                        exp_pred_raw = expert_preds_raw[i].item()
                        exp_pred_remapped = expert_prediction_map_inv[exp_pred_raw]
                        exp_confidence = expert_confidences[i].item()

                        true_gatekeeper_label = 1 if true_label == 2 else 0
                        final_pred = -1 # Initialize

                        current_file_path = val_files_fold[batch_idx * args.batch_size + i]

                        cekl_feature = cekl_features[i].cpu().numpy() if cekl_features is not None else None
                        for j,(key, value) in enumerate(cekl_features_dict.items()):
                            cekl_features_dict[key] = cekl_feature[j] if cekl_feature is not None else None

                        # --- PIPELINE LOGIC AND ERROR ATTRIBUTION ---
                        if gk_pred == 1: # Gatekeeper says "IS_SGMSE"
                            final_pred = 2
                            if true_gatekeeper_label == 1: # CORRECT decision
                                gatekeeper_correct_confidence.append(gk_confidence)
                            else: # WRONG decision (False Positive)
                                sample_info = {
                                    'file_path': val_files_fold[batch_idx * args.batch_size + i],
                                    'true_label': true_label,
                                    'predicted_label': final_pred,                                    
                                }
                                sample_info.update(cekl_features_dict) # Add CE/KL features
                                gatekeeper_false_positives.append(sample_info)

                        else: # Gatekeeper says "NOT_SGMSE", route to expert
                            final_pred = exp_pred_remapped
                            
                            # Store expert result REGARDLESS of correctness for its own performance report
                            expert_results_on_routed_data.append({'true': true_label, 'pred': exp_pred_remapped})
                            
                            if true_gatekeeper_label == 0: # CORRECT routing decision
                                if final_pred == true_label: # And expert was also correct
                                    expert_correct_confidence.append(exp_confidence)
                                    file_info = {
                                        'file_path': current_file_path,
                                        'true_label': true_label, # e.g., 0, 1, or 3
                                        'predicted_label': final_pred, # will be same as true_label
                                        'expert_confidence': exp_confidence
                                    }
                                    file_info.update(cekl_features_dict) # Add CE/KL features
                                    expert_correct_samples.append(file_info)
                                else: # Gatekeeper was right, but Expert was wrong
                                    file_info = {
                                        'file_path': val_files_fold[batch_idx * args.batch_size + i],
                                        'true_label': true_label,
                                        'predicted_label': final_pred
                                    }
                                    file_info.update(cekl_features_dict) # Add CE/KL features
                                    expert_errors.append(file_info)
                            else: # WRONG routing decision (False Negative)
                                file_info = {
                                    'file_path': val_files_fold[batch_idx * args.batch_size + i],
                                    'true_label': true_label, # This will always be 2
                                    'predicted_label': final_pred
                                }
                                file_info.update(cekl_features_dict) # Add CE/KL features
                                gatekeeper_false_negatives.append(file_info)

                        all_final_preds.append(final_pred)

                        # "Hardest Sample" analysis based on Gatekeeper uncertainty
                        uncertainty = abs(gk_confidence - 0.5)
                        if uncertainty < 0.1: # Threshold for "low confidence"
                            file_info = {
                                'file_path': val_files_fold[batch_idx * args.batch_size + i],
                                'true_label': true_label,
                                'gatekeeper_pred_label': 1 if gk_pred == 1 else 0,
                                'gatekeeper_confidence': gk_confidence,
                                'uncertainty_score': uncertainty
                            }
                            file_info.update(cekl_features_dict)
                            low_confidence_samples.append(file_info)


            # --- FINAL METRICS AND ANALYSIS ---
            print("\n--- Overall Pipeline Performance ---")
            print(classification_report(all_true_labels, all_final_preds, target_names=list(global_best_model_to_label_dict.keys()), zero_division=0))
            conf_mat = confusion_matrix(all_true_labels, all_final_preds, labels=list(global_best_model_to_label_dict.values()))
            print(f"Fold {fold+1} Confusion Matrix:\n{conf_mat}")
            
            # --- Deeper Analysis using Helper Functions ---
            total_errors = len(gatekeeper_false_positives) + len(gatekeeper_false_negatives) + len(expert_errors)
            
            # 1. Analyze Gatekeeper's standalone performance
            recommender_pipeline.analyze_gatekeeper_performance(all_true_labels, all_gatekeeper_preds)

            # 2. Analyze Expert's performance on the data it received
            print("\n--- Expert Performance Analysis (on data routed by Gatekeeper) ---")
            if expert_results_on_routed_data:
                # Prepare labels for a 3-class classification report
                expert_true_labels = [expert_label_map[d['true']] for d in expert_results_on_routed_data if d['true'] != 2]
                expert_pred_labels = [expert_label_map[d['pred']] for d in expert_results_on_routed_data if d['true'] != 2]
                print(classification_report(
                    expert_true_labels,
                    expert_pred_labels,
                    target_names=['All Failed', 'CDiffuSE', 'StoRM'], # Adjust as needed
                    labels=[0, 1, 2],
                    zero_division=0
                ))
            else:
                print("No samples were routed to the Expert model.")

            # 3. Analyze the contribution of each component to total errors
            recommender_pipeline.analyze_error_contribution(
                gatekeeper_fp_count=len(gatekeeper_false_positives),
                gatekeeper_fn_count=len(gatekeeper_false_negatives),
                expert_error_count=len(expert_errors),
                total_errors=total_errors
            )

            # 4. Analyze confidence of correctly classified samples
            recommender_pipeline.analyze_correctly_classified(
                gatekeeper_correct_confidence,
                expert_correct_confidence
            )

            # 5. Report on low-confidence samples
            if low_confidence_samples:
                write_misclassified_labels_to_csv(
                    low_confidence_samples, 
                    f'gatekeeper_low_confidence_samples_fold{fold+1}.csv'
                )
                print(f"\nFold {fold+1}: Found and saved {len(low_confidence_samples)} low-confidence Gatekeeper samples.")
            if expert_correct_samples:
                # You can reuse your existing CSV writing function
                write_misclassified_labels_to_csv(
                    expert_correct_samples, 
                    f'expert_correctly_classified_samples_fold{fold+1}.csv'
                )
                print(f"\nFold {fold+1}: Saved {len(expert_correct_samples)} correctly classified Expert samples to CSV.")
            else:
                print("\nFold {fold+1}: No correctly classified samples by the Expert were recorded.")


    # --- K-Fold Summary ---
    if fold_results:
        print("\n--- K-Fold Cross-Validation Summary ---")
        avg_accuracy = np.mean([res['accuracy'] for res in fold_results])
        avg_f1 = np.mean([res['f1'] for res in fold_results])
        avg_precision = np.mean([res['precision'] for res in fold_results])
        avg_recall = np.mean([res['recall'] for res in fold_results])
        print(f'Average Val Accuracy: {avg_accuracy:.4f}')
        print(f'Average Val F1 Score: {avg_f1:.4f}')
        print(f'Average Val Precision: {avg_precision:.4f}')
        print(f'Average Val Recall: {avg_recall:.4f}')
        # save logs as txt
        with open(f'k_fold_summary_{args.feature_type}_{args.pretrained_models}.txt', 'w') as f:
            f.write("K-Fold Cross-Validation Summary:\n")
            f.write(f'Average Val Accuracy: {avg_accuracy:.4f}\n')
            f.write(f'Average Val F1 Score: {avg_f1:.4f}\n')
            f.write(f'Average Val Precision: {avg_precision:.4f}\n')
            f.write(f'Average Val Recall: {avg_recall:.4f}\n')
            for res in fold_results:
                f.write(f"Fold {res['fold']}: Acc: {res['accuracy']:.4f}, F1: {res['f1']:.4f}, "
                        f"Prec: {res['precision']:.4f}, Rec: {res['recall']:.4f}\n")
    else:
        print("No K-Fold results to summarize.")