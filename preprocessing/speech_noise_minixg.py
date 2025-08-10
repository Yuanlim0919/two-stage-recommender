import os
from glob import glob
from librosa import load
from librosa.core import resample
import argparse
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from soundfile import write
from tqdm import tqdm
import torchaudio
import torchaudio.functional as F
import torch


# Python script for generating noisy mixtures for training
#
# Mix crema with CHiME3 noise with SNR sampled uniformly in [min_snr, max_snr]


min_snr = 0
max_snr = 20
sr = 16000

def generate_noise_audio(noise_file_path, target_file):
    noise, _ = torchaudio.load(noise_file_path)
    min_snr = 1
    max_snr = 10
    snr_range = torch.tensor([np.random.randint(min_snr, max_snr)])
    if noise.shape[1] < target_file.shape[1]:
        noise = torch.nn.functional.pad(noise, (0, target_file.shape[1] - noise.shape[1]))
    elif noise.shape[1] > target_file.shape[1]:
        noise = noise[:,:target_file.shape[1]]
    noisy_audio  = F.add_noise(target_file, noise, snr=snr_range)
    return noisy_audio


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--speech_dir", type=str, help='path to Crema directory')
    parser.add_argument("--noise_dir", type=str,  help='path to MS-SNSD directory')
    parser.add_argument("--target_dir", type=str, help='target path for training files')
    args = parser.parse_args()

    # Clean speech for training
    speech_files = sorted(glob(args.speech_dir + '**/*.wav', recursive=True))
    sample_rate = 16000
    
    train_speech_files = speech_files

    noise_files = glob(args.noise_dir + '**/*.wav', recursive=True)

    # Create target dir
    train_clean_path = Path(os.path.join(args.target_dir, 'clean'))
    train_noisy_path = Path(os.path.join(args.target_dir, 'noisy'))

    train_clean_path.mkdir(parents=True, exist_ok=True)
    train_noisy_path.mkdir(parents=True, exist_ok=True)

    # Initialize seed for reproducability
    np.random.seed(0)

    # Create files for training
    print('Create training files')
    for i, speech_file in enumerate(tqdm(train_speech_files)):
        speech, _ = torchaudio.load(speech_file)
        noise_file = np.random.choice(noise_files)
        noisy_audio = generate_noise_audio(noise_file, speech)
        speech_fname = speech_file.split('/')[-1].split('.')[0].split('_')[0]
        noise_fname = noise_file.split('/')[-1].split('.')[0].split('_')[0]

        torchaudio.save(os.path.join(train_clean_path, f'{speech_fname}_{noise_fname}.wav'), speech, sample_rate)
        torchaudio.save(os.path.join(train_noisy_path, f'{speech_fname}_{noise_fname}.wav'), noisy_audio, sample_rate)