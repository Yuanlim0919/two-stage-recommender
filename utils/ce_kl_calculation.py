import argparse
import os
from pathlib import Path
import torch
import torchaudio
import numpy as np
from tqdm import tqdm # A great library for progress bars

# You'll need to import your NoiseDataPreprocessor
from ce_kl_feature_extraction import NoiseDataPreprocessor 

def preprocess_and_save_features(audio_dir, output_dir, target_sr=16000):
    """
    Calculates CE/KL features for all .wav files in a directory
    and saves them to a parallel directory structure.
    """
    print(f"Processing directory: {audio_dir}")
    
    # Define your feature extraction parameters here
    # These must be consistent with what you used in the dataset before
    ndp_n_fft = 1024 
    ndp_hop_length = 512
    
    # Initialize your preprocessors
    noise_data_preprocessor = NoiseDataPreprocessor(
        audio_paths=audio_dir, # This is just for init, not critical
        sr=target_sr, n_fft=ndp_n_fft, hop_length=ndp_hop_length
    )
    stft_transform_for_cekl = torchaudio.transforms.Spectrogram(
        n_fft=ndp_n_fft, hop_length=ndp_hop_length, power=2.0
    )

    audio_files = list(Path(audio_dir).glob("*.wav"))
    
    # Create the output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for audio_path in tqdm(audio_files, desc=f"Processing {os.path.basename(audio_dir)}"):
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # --- PREPARE WAVEFORM ---
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sample_rate != target_sr:
                waveform = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=target_sr
                )(waveform)
            
            # --- CALCULATE FEATURES ---
            s_power_torch = stft_transform_for_cekl(waveform)
            s_power_numpy = s_power_torch.squeeze(0).numpy()
            cekl_features_numpy = noise_data_preprocessor.extract_features_by_sample(
                s_power_numpy, target_sr
            )
            
            # --- SAVE FEATURES ---
            # Construct the output path. Change .wav to .npy
            output_filename = audio_path.stem + ".npy"
            output_filepath = Path(output_dir) / output_filename
            
            # Save the numpy array to disk
            np.save(output_filepath, cekl_features_numpy)

        except Exception as e:
            print(f"Failed to process {audio_path}: {e}")

if __name__ == '__main__':
    # This allows you to run this script from the command line
    parser = argparse.ArgumentParser(description="Pre-calculate CE/KL features for audio datasets.")
    parser.add_argument('--base_data_dir', type=str, required=True, help="Base directory of your audio datasets.")
    parser.add_argument('--output_base_dir', type=str, required=True, help="Base directory to save the features.")
    
    args = parser.parse_args()

    # List of your dataset subdirectories
    dataset_subdirs = [
        'noisy_blind_testset_v3_challenge_withSNR_16k',
        'voicebank/noisy_testset_wav',
        'crema_mssnsd_non_stationary/test/noisy',
        'sgmse/noisy_trainset_28spk_wav',
        'ljspeech_esc50/noisy'
    ]
    
    for subdir in dataset_subdirs:
        audio_dir = os.path.join(args.base_data_dir, subdir)
        output_dir = os.path.join(args.output_base_dir, subdir)
        
        if os.path.isdir(audio_dir):
            preprocess_and_save_features(audio_dir, output_dir)
        else:
            print(f"Skipping non-existent directory: {audio_dir}")

    print("All features pre-calculated and saved.")