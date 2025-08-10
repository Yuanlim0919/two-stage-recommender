import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def analyze_audio(audio_path):
    """
    Analyzes a single audio file to compute its mel spectrogram and spectral entropy.

    Args:
        audio_path (str): The path to the audio file.

    Returns:
        dict: A dictionary containing the analysis results.
    """
    print(f"Analyzing {os.path.basename(audio_path)}...")
    try:
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Could not load audio file {audio_path}: {e}")
        return None

    # Define STFT parameters
    hop_length = 512
    n_fft = 2048

    # Compute Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )

    # --- Compute Spectral Entropy ---
    # Normalize the mel spectrogram across frequency bins for each time frame
    mel_spec_norm = mel_spectrogram / (np.sum(mel_spectrogram, axis=0, keepdims=True) + 1e-10)
    # Compute spectral entropy using log base 2 for units of "bits"
    spectral_entropy = -np.sum(mel_spec_norm * np.log2(mel_spec_norm + 1e-10), axis=0)

    # --- Create Time Vector for plotting ---
    times = librosa.frames_to_time(
        np.arange(spectral_entropy.shape[0]),
        sr=sr,
        hop_length=hop_length
    )
    
    # Return all computed data in a dictionary
    return {
        "audio_path": audio_path,
        "mel_spectrogram": mel_spectrogram,
        "spectral_entropy": spectral_entropy,
        "times": times,
        "sr": sr,
        "hop_length": hop_length,
    }

def plot_stacked_analysis(analyses, output_filename="demo_plot.png"):
    """
    Creates a vertically stacked plot for a list of audio analyses.

    Args:
        analyses (list): A list of dictionaries, where each dict is the result
                         from the analyze_audio function.
        output_filename (str): The name of the file to save the plot.
    """
    num_files = len(analyses)
    if num_files == 0:
        print("No analyses to plot.")
        return

    # Create a figure with 2 rows for each file, and 1 column.
    # The height of the figure is scaled by the number of files.
    fig, axs = plt.subplots(
        nrows=num_files, #  * 2
        ncols=1, 
        figsize=(12, 5 * num_files),
        # Share the X-axis between each spectrogram and its entropy plot
        gridspec_kw={'height_ratios': [2] * (num_files)} # Adjust ratios if needed
    )

    for i, analysis in enumerate(analyses):
        # Determine the correct axes for the current file
        ax_spec = axs[i] #*2
        # ax_entropy = axs[i * 2 + 1]

        # Get a short, clean name for the title
        file_title = os.path.basename(analysis['audio_path'])

        # --- Plot 1: Mel Spectrogram ---
        img = librosa.display.specshow(
            librosa.power_to_db(analysis['mel_spectrogram'], ref=np.max),
            sr=analysis['sr'],
            hop_length=analysis['hop_length'],
            x_axis='time',
            y_axis='mel',
            ax=ax_spec
        )
        # fig.colorbar(img, ax=ax_spec, format='%+2.0f dB')
        ax_spec.set_title(f"Mel Spectrogram: {file_title}")
        ax_spec.set_xlabel('') # Hide x-label to avoid clutter
        ax_spec.set_ylabel('Hz')

        # --- Plot 2: Spectral Entropy ---
        # ax_entropy.plot(analysis['times'], analysis['spectral_entropy'])
        # ax_entropy.set_title(f"Spectral Entropy: {file_title}")
        # ax_entropy.set_ylabel('Entropy (bits)')
        # ax_entropy.grid(True, linestyle='--', alpha=0.6)
        # ax_entropy.set_xlim([0, analysis['times'][-1]]) # Set x-limit for this plot
        
        # Only show the x-label ("Time (s)") on the very last plot
        if i == num_files - 1:
            ax_spec.set_xlabel('Time (s)')
        # else:
        #     ax_entropy.set_xlabel('')


    plt.tight_layout(pad=2.0) # Add some padding between plots
    plt.savefig(output_filename)
    print(f"\nStacked plot saved to {output_filename}")
    plt.show()


if __name__ == "__main__":
    # --- Define the list of audio files you want to process ---
    # Using librosa's built-in examples so the code runs out-of-the-box
    audio_files = [
        '/success_fail_estimation/noisy_blind_testset_v3_challenge_withSNR_16k/ms_realrec_english_male_SNR_24.39dB_webcam_A28O8I1SYFZO7A_chipbag_near_5.wav',
        '/success_fail_estimation/noisy_blind_testset_v3_challenge_withSNR_16k/ms_realrec_emotional_female_SNR_21.41dB_speaker_ASEW6NZHLI41K_breathing_far_surprise_1.wav',
        '/success_fail_estimation/noisy_blind_testset_v3_challenge_withSNR_16k/ms_realrec_emotional_female_SNR_19.77dB_laptopmic_A3VBNWON5XOUVS_breathing_near_si2322.wav',
        '/success_fail_estimation/noisy_blind_testset_v3_challenge_withSNR_16k/nonenglish_tonal_synthetic_male_SNR_11.84dB_vietnamese_4.wav'
    ]

    # Generate white noise for comparison
    # sr_noise = 22050
    # duration = 5
    # y_noise = np.random.randn(sr_noise * duration)
    # noise_path = "temp_white_noise.wav"
    # import soundfile as sf
    # sf.write(noise_path, y_noise, sr_noise)
    # audio_files.append(noise_path)
    
    # --- Process all files ---
    all_analyses = []
    for f_path in audio_files:
        result = analyze_audio(f_path)
        if result:
            all_analyses.append(result)
    
    # Clean up the temporary noise file
    # if os.path.exists(noise_path):
    #     os.remove(noise_path)

    # --- Plot all results together ---
    if all_analyses:
        plot_stacked_analysis(all_analyses)