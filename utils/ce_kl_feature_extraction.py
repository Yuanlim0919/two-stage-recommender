import os
import numpy as np
import librosa
import warnings
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class NoiseDataPreprocessor:
    def __init__(self, audio_paths, sr=22050, n_fft=2048, hop_length=512):
        if audio_paths:
            self.audio_paths = os.listdir(audio_paths)
            self.audio_paths = [os.path.join(audio_paths, path) for path in self.audio_paths]
        # self.audio_paths = self.audio_paths[:5000]
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.features = []
        self.noise_category_dict = {
            # 0 means statioanry noise, 1 means non-stationary noise
            'Field': 0,
            'AirConditioner': 0,
            'AirportAnnouncement': 1,
            'AirportAnnouncements': 1,
            'Babble': 1,
            'Bus': 1,
            'Cafe':1,
            'CafeTeria': 1,
            'Car':0,
            'CopyMachine': 1,
            'Hallway': 1,
            'Kitchen': 1,
            'LivingRoom': 1,
            'Metro': 1,
            'Munching': 1,
            'NeighborSpeaking': 1,
            'Neighbor':1,
            'Office': 1,
            'Park': 1,
            'Restaurant': 1,
            'ShuttingDoor': 1,
            'Square': 1,
            'SqueakyChair': 1,
            'Station': 1,
            'Traffic': 1,
            'Typing': 1,
            'VacuumCleaner': 0,
            'WasherDryer':1,
            'Washing':1
        }
        pass

    def spectral_cross_entropy(self,
                               S: np.ndarray,
                           sr: int,
                           n_fft: int = 2048,
                           hop_length: int = 512,
                           eps: float = 1e-10,
                           kl_div = True) -> np.ndarray:
        """
        Computes the pairwise spectral cross-entropy between all time frames
        of a signal's spectrogram.

        The cross-entropy H(P, Q) between two probability distributions P and Q
        is calculated as: H(P, Q) = - Σ P(i) * log2(Q(i)).

        This function first calculates the power spectrogram, then normalizes each
        time frame (column) to sum to 1, treating it as a probability distribution
        (PMF) of energy across frequency bins. It then computes the cross-entropy
        between every pair of frames.

        Args:
            signal: The input audio time series (1D numpy array).
            sr: The sampling rate of the signal.
            n_fft: The FFT window size.
            hop_length: The hop length for the STFT.
            eps: A small epsilon value added for numerical stability (to avoid log(0)
                and division by zero).

        Returns:
            A 2D numpy array `ce_matrix` of shape (n_frames, n_frames), where
            `n_frames` is the number of time frames in the spectrogram.
            `ce_matrix[i, j]` contains the cross-entropy H(P_i, Q_j), where P_i
            is the normalized spectral distribution of frame `i`, and Q_j is the
            normalized spectral distribution of frame `j`.
            Returns an empty array if the signal results in an empty spectrogram.

        Note:
            - Low cross-entropy H(P_i, Q_j) indicates that the spectral shape of
            frame j is a good predictor for the spectral shape of frame i.
            - H(P, Q) is not symmetric, so ce_matrix[i, j] != ce_matrix[j, i] generally.
            - The diagonal elements ce_matrix[i, i] represent the negative entropy
            of frame i, H(P_i, P_i) = - Σ P_i(f) * log2(P_i(f)), which is the
            standard spectral entropy (negated).
            - For measuring dissimilarity, KL Divergence D_KL(P_i || Q_j) might be
            more appropriate: D_KL(P_i || Q_j) = H(P_i, Q_j) - H(P_i), where H(P_i)
            is the spectral entropy of frame i.
        """
        # 1. Compute the magnitude squared (power) spectrum
        # try:
        #     S = np.abs(librosa.stft(y=signal, n_fft=n_fft, hop_length=hop_length))**2
        #     if S.shape[1] == 0:
        #         warnings.warn("Input signal resulted in an empty spectrogram.")
        #         return np.array([[]]) # Return empty 2D array consistent with shape idea
        # except Exception as e:
        #     warnings.warn(f"Librosa STFT failed: {e}")
        #     return np.array([[]])

        n_freq_bins, n_frames = S.shape

        # 2. Normalize the power spectrum columns to get PMFs
        # Sum power across frequency bins for each frame
        frame_power_sum = np.sum(S, axis=0, keepdims=True)

        # Handle frames with zero power to avoid NaN (replace sum with 1 to yield 0 after division)
        frame_power_sum[frame_power_sum < eps] = 1.0

        # Normalize each frame (column) to sum to 1
        psd = S / (frame_power_sum + eps) # Add eps for general numerical stability

        # Add epsilon *before* log to handle zero probabilities within frames
        psd_stable = psd + eps

        # 3. Compute pairwise cross-entropy matrix
        # Initialize matrix: rows correspond to P, columns correspond to Q in H(P, Q)
        cross_entropy_matrix = np.zeros((n_frames, n_frames))

        # Efficiently calculate using broadcasting
        # We want ce_matrix[i, j] = H(P_i, Q_j) = - sum( P_i(f) * log2(Q_j(f)) )

        log2_psd_stable = np.log2(psd_stable) # Precompute log2(Q_j(f)) for all j

        for i in range(n_frames):
            # Select the i-th frame's distribution (P_i)
            # Keep it as a column vector (n_freq_bins, 1) for broadcasting
            p_i = psd[:, i:i+1] # This is P_i

            # Calculate H(P_i, Q_j) for all j simultaneously
            # Element-wise product: p_i(f) * log2(Q_j(f)) (broadcasts p_i across columns of log2_psd)
            # Sum over frequency axis (axis=0)
            # Result is a row vector (1, n_frames) containing H(P_i, Q_j) for fixed i, varying j
            h_pi_qj = -np.sum(p_i * log2_psd_stable, axis=0)

            # Store the result in the i-th row of the matrix
            cross_entropy_matrix[i, :] = h_pi_qj
        
        if kl_div:
            # --- Step 4: Compute Shannon Entropy H(P_i) for each frame i ---
            # H(P_i) = - Σ P_i(f) * log2(P_i(f))
            # We can calculate this directly or use the diagonal of the cross-entropy matrix.
            # Note: cross_entropy_matrix[i, i] = H(P_i, P_i) = - Σ P_i(f) * log2(P_i(f)) = H(P_i)
            # So, the diagonal directly gives H(P_i). Let's use that for consistency.
            shannon_entropies_h_pi = np.diag(cross_entropy_matrix).copy() # Shape (n_frames,)

            # --- Step 5: Compute KL Divergence D_KL(P_i || Q_j) ---
            # D_KL(P_i || Q_j) = H(P_i, Q_j) - H(P_i)
            # We need to subtract H(P_i) from each element in the i-th row of H(P_i, Q_j).
            # Reshape H(P_i) to allow broadcasting (n_frames, 1)
            h_pi_reshaped = shannon_entropies_h_pi.reshape(-1, 1)

            # KL = H(P,Q) - H(P)
            kl_matrix = cross_entropy_matrix - h_pi_reshaped

            # Ensure non-negativity due to potential floating point inaccuracies near zero
            kl_matrix[kl_matrix < 0] = 0.0
            return cross_entropy_matrix, kl_matrix

    def get_mean_vars(self, feature_matrix, matrix_type='unknown'):
        """Calculates summary stats from a CE or KL matrix.

        Args:
            feature_matrix: The n_frames x n_frames matrix (CE or KL).
            matrix_type: String ('ce' or 'kl') to indicate matrix type.
                        Controls whether diagonal variance is calculated.

        Returns:
            A list of features. Includes diagonal variance only for 'ce'.
        """
        features = []
        n_frames = feature_matrix.shape[0]

        # --- Diagonal Variance (only for CE matrix) ---
        if matrix_type == 'ce':
            var_diag = np.var(np.diag(feature_matrix)) if n_frames > 0 else 0
            features.append(var_diag)
        # For KL matrix, var(diag) is ~0, so we don't add it,
        # or we could explicitly add 0.0 if a fixed length is crucial here.
        # Let's assume we just omit it for KL for now.

        # --- Off-Diagonal Stats ---
        if n_frames >= 2:
            off_diagonal_mask = ~np.eye(n_frames, dtype=bool)
            off_diag_values = feature_matrix[off_diagonal_mask]
            mean_off_diag = np.mean(off_diag_values)
            variance_off_diag = np.var(off_diag_values)
        else: # Handle case n_frames < 2
            mean_off_diag = 0.0
            variance_off_diag = 0.0
        features.extend([mean_off_diag, variance_off_diag])

        # --- Adjacent Frame Stats ---
        if n_frames >= 2:
            diag_p1 = np.diag(feature_matrix, k=1)
            diag_n1 = np.diag(feature_matrix, k=-1)
            # Ensure concatenation only happens if there are elements
            adj_values = np.concatenate((diag_p1, diag_n1))
            mean_adj = np.mean(adj_values) if adj_values.size > 0 else 0.0
            var_adj = np.var(adj_values) if adj_values.size > 0 else 0.0
        else: # Handle case n_frames < 2
            mean_adj = 0.0
            var_adj = 0.0
        features.extend([mean_adj, var_adj])

        return features
    
    def get_spectral_features(self,
                          S_mag: np.ndarray,
                          sr: int,
                          n_fft: int = 2048,
                          hop_length: int = 512,
                          n_bands_contrast: int = 6,
                          rolloff_percent: float = 0.85) -> np.ndarray:
        """
        Calculates a set of standard spectral features from an audio signal,
        focusing on their mean and variance over time to capture temporal dynamics.

        Features Calculated (Mean and Variance over time):
            - Spectral Centroid
            - Spectral Bandwidth
            - Spectral Contrast (per band)
            - Spectral Flatness
            - Spectral Rolloff
            - Zero-Crossing Rate

        Args:
            signal: The input audio time series (1D numpy array).
            sr: The sampling rate of the signal.
            n_fft: The FFT window size for STFT.
            hop_length: The hop length for STFT.
            n_bands_contrast: Number of frequency bands for spectral contrast.
            rolloff_percent: The roll-off percentage for spectral rolloff.

        Returns:
            A 1D numpy array containing the calculated features (mean and variance
            for each base feature type). Returns an empty array if the signal is
            too short to produce features.
        """
        feature_vector = []

        try:
            # --- Calculate STFT and Power Spectrogram ---
            #S_mag = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))
            S_power = S_mag**2

            n_frames = S_power.shape[1]

            if n_frames < 1:
                warnings.warn("Signal too short to produce STFT frames.")
                return np.array([]) # Return empty if no frames

            # Helper function to calculate mean and variance, handling few frames
            def _mean_var(data):
                mean = np.mean(data) if data.size > 0 else 0.0
                var = np.var(data) if data.size >= 2 else 0.0 # Variance requires at least 2 points
                return mean, var

            # --- Spectral Centroid ---
            # cent = librosa.feature.spectral_centroid(S=S_power)[0]
            # mean_cent, var_cent = _mean_var(cent)
            # feature_vector.extend([mean_cent, var_cent])

            # # --- Spectral Bandwidth ---
            # bw = librosa.feature.spectral_bandwidth(S=S_power)[0]
            # mean_bw, var_bw = _mean_var(bw)
            # feature_vector.extend([mean_bw, var_bw])

            # --- Spectral Contrast ---
            # Shape: (n_bands + 1, n_frames)
            contrast = librosa.feature.spectral_contrast(S=S_power, n_bands=n_bands_contrast)
            for i in range(contrast.shape[0]): # Iterate through each contrast band
                mean_contrast_band, var_contrast_band = _mean_var(contrast[i, :])
                feature_vector.extend([mean_contrast_band, var_contrast_band])

            # --- Spectral Flatness ---
            # flatness = librosa.feature.spectral_flatness(S=S_power)[0]
            # mean_flatness, var_flatness = _mean_var(flatness)
            # feature_vector.extend([mean_flatness, var_flatness])

            # # --- Spectral Rolloff ---
            # rolloff = librosa.feature.spectral_rolloff(S=S_power, roll_percent=rolloff_percent)[0]
            # mean_rolloff, var_rolloff = _mean_var(rolloff)
            # feature_vector.extend([mean_rolloff, var_rolloff])

            # --- Zero-Crossing Rate ---
            # # Calculated frame-wise on the time-domain signal
            # zcr = librosa.feature.zero_crossing_rate(y=signal, frame_length=n_fft, hop_length=hop_length)[0]
            # # Ensure zcr length matches n_frames (librosa padding might differ slightly)
            # if len(zcr) > n_frames:
            #     zcr = zcr[:n_frames]
            # elif len(zcr) < n_frames:
            #     # Pad with the last value if shorter, though less likely
            #     zcr = np.pad(zcr, (0, n_frames - len(zcr)), mode='edge')

            # mean_zcr, var_zcr = _mean_var(zcr)
            # feature_vector.extend([mean_zcr, var_zcr])

        except Exception as e:
            warnings.warn(f"Error during spectral feature extraction: {e}")
            return np.array([]) # Return empty array on error

        return np.array(feature_vector)
    
    def extract_features_by_sample(self,signal, sr):
        # if len(signal) < 20 * sr:
        #     # Replicate the signal to make it at least 20 seconds long
        #     signal = np.tile(signal, int(np.ceil(20 * sr / len(signal))))
        #     # Trim to 20 seconds
        #     signal = signal[:20 * sr]
        # else:
        # # first 20 seconds
        #     signal = signal[:20 * sr]

        # Compute the spectral cross-entropy matrix
        ce_matrix, kl_matrix = self.spectral_cross_entropy(signal, sr)
        # Get the mean and variance of the off-diagonal elements
        ce_stats = self.get_mean_vars(ce_matrix, matrix_type='ce') # Now 5 features
        kl_stats = self.get_mean_vars(kl_matrix, matrix_type='kl') # Now 4 features
        # spectral_features = self.get_spectral_features(signal, sr) # Now 12 features
        # Combine the features
        combined_features = ce_stats + kl_stats  # Total 19 features
        return np.array(combined_features)

    
    def extract_features(self):
        # Iterate through the audio files
        for i, audio_path in tqdm(enumerate(self.audio_paths)):
            if audio_path.endswith('.wav'):
                noise_category = audio_path.split('/')[-1].split('_')[-2]
                noise_category = self.noise_category_dict[noise_category]
            # Load the audio file
            signal, sr = librosa.load(audio_path, sr=self.sr)
            # Check if the signal is too short, if so replicate it
            if len(signal) < 20 * sr:
                # Replicate the signal to make it at least 20 seconds long
                signal = np.tile(signal, int(np.ceil(20 * sr / len(signal))))
                # Trim to 20 seconds
                signal = signal[:20 * sr]
            else:
            # first 20 seconds
                signal = signal[:20 * sr]

            # Compute the spectral cross-entropy matrix
            ce_matrix, kl_matrix = self.spectral_cross_entropy(signal, sr)
            # Get the mean and variance of the off-diagonal elements
            ce_stats = self.get_mean_vars(ce_matrix, matrix_type='ce') # Now 5 features
            kl_stats = self.get_mean_vars(kl_matrix, matrix_type='kl') # Now 4 features
            # spectral_features = self.get_spectral_features(signal, sr) # Now 12 features
            # Combine the features
            combined_features = ce_stats + kl_stats  # Total 19 features
            self.features.append(combined_features + [noise_category])
        return np.array(self.features)
    
    def process_audio_file(self, 
                    audio_path, 
                    sr, 
                    noise_category_dict, 
                    get_mean_vars_func, 
                    spectral_ce_kl_func, 
                    # spectral_features_func,
                    dataset_name
                    ):
        """Processes a single audio file to extract features."""
        try:
            if dataset_name != "crema_mssnsd":
                noise_category = 0
            if audio_path.endswith('.wav') :
                # --- Label Extraction ---
                # Adjust parsing based on your exact filename structure if needed
                if dataset_name == "crema_mssnsd":
                    parts = audio_path.split('/')[-1].split('_')
                    noise_category_str = parts[-2] if len(parts) >= 2 else None
                    if noise_category_str is None or noise_category_str not in noise_category_dict:
                        warnings.warn(f"Could not determine category for {audio_path}. Skipping.")
                        return None
                    noise_category = noise_category_dict[noise_category_str]

                # --- Load Audio & Prepare Signal ---
                signal, loaded_sr = librosa.load(audio_path, sr=sr)
                target_len = 20 * sr
                current_len = len(signal)

                if current_len == 0:
                    warnings.warn(f"Signal is empty for {audio_path}. Skipping.")
                    return None

                if current_len < target_len:
                    # Replicate the signal efficiently
                    n_repeats = int(np.ceil(target_len / current_len))
                    signal = np.tile(signal, n_repeats)[:target_len]
                elif current_len > target_len:
                    signal = signal[:target_len]

                # --- Calculate STFT ONCE ---
                S_mag = np.abs(librosa.stft(signal, n_fft=2048, hop_length=512))
                S_power = S_mag**2

                if S_power.shape[1] < 2: # Need at least 2 frames for some stats
                    warnings.warn(f"Signal too short after STFT for {audio_path} (< 2 frames). Skipping.")
                    return None

                # --- Compute CE/KL Matrices from S_power ---
                # Assumes this function now accepts S_power and returns both
                ce_matrix, kl_matrix = spectral_ce_kl_func(S_power, sr) # Pass S_power

                if ce_matrix is None or kl_matrix is None: # Add check if function can fail
                    warnings.warn(f"CE/KL matrix calculation failed for {audio_path}. Skipping.")
                    return None
                
                # --- Spectral Features (Optional, if uncommented) ---
                # spectral_features = spectral_features_func(S_power, sr) # Pass S_power
                # if spectral_features.size == 0:
                #     warnings.warn(f"Spectral feature extraction failed for {audio_path}. Skipping.")

                # --- Get Summary Stats ---
                ce_stats = self.get_mean_vars(ce_matrix, matrix_type='ce')
                kl_stats = self.get_mean_vars(kl_matrix, matrix_type='kl')

                # --- Combine Features ---
                combined_features = ce_stats + kl_stats # + spectral_features.tolist() # If spectral included
                # combined_features = ce_stats + kl_stats + spectral_features.tolist() 
                # combined_features = spectral_features.tolist() # If spectral included

                return combined_features + [noise_category]
            else:
                return None # Skip non-wav files

        except Exception as e:
            warnings.warn(f"Error processing {audio_path}: {e}")
            return None

    def extract_features_parallel(self, n_jobs=-1, dataset_name='crema_mssnsd'):
        """
        Extracts features from audio files in parallel.

        Args:
            n_jobs (int): Number of CPU cores to use. -1 means use all available.

        Returns:
            np.ndarray: Array of features with labels in the last column.
        """
        print(f"Starting feature extraction for {len(self.audio_paths)} files using {n_jobs} jobs...")

        # Use joblib.Parallel to process files
        # Pass necessary methods/data that the worker function needs
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(self.process_audio_file)(
                audio_path,
                self.sr,
                self.noise_category_dict,
                self.get_mean_vars,             # Pass the method itself
                self.spectral_cross_entropy,    # Pass the modified method
                # self.get_spectral_features,      # Pass the modified method (if used)
                dataset_name
            ) for audio_path in tqdm(self.audio_paths, desc="Processing files")
        )

        # Filter out None results (from skipped files or errors)
        valid_results = [res for res in results if res is not None]

        if not valid_results:
            print("Warning: No valid features were extracted.")
            return np.array([])

        print(f"Successfully extracted features from {len(valid_results)} files.")
        return np.array(valid_results)

    def train_valid_test_split(self, features, labels, test_size=0.2, valid_size=0.1):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size)
        # Further split the training set into training and validation sets
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size)
        return X_train, X_valid, X_test, y_train, y_valid, y_test
        pass


def prepare_feature_standard_scaler_by_fold(dataset_to_scale: Dataset, batch_size_for_extraction: int = 32, num_workers_for_extraction: int = 0):
    """
    Prepare a StandardScaler fitted on all features from the given Dataset.
    Assumes features are suitable for StandardScaler (e.g., CE/KL features).
    """
    from sklearn.preprocessing import StandardScaler
    
    # Use a DataLoader for potentially faster feature extraction, especially with num_workers > 0
    # Ensure this temp DataLoader doesn't shuffle if order matters for some reason (not for fitting scaler)
    temp_loader = DataLoader(dataset_to_scale, batch_size=batch_size_for_extraction, 
                             shuffle=False, num_workers=num_workers_for_extraction)
    
    all_features_list = []
    print(f"Extracting features from {len(dataset_to_scale)} samples to fit StandardScaler...")
    for features_batch, _ in tqdm(temp_loader, desc="Fitting Scaler"):
        # Assuming features_batch is already a tensor on CPU or can be moved
        all_features_list.append(features_batch.cpu().numpy())

    if not all_features_list:
        warnings.warn("No features extracted. Scaler cannot be fitted.")
        return None

    # Concatenate all batches into a single NumPy array
    # all_features_np should be (total_samples, num_features)
    try:
        all_features_np = np.concatenate(all_features_list, axis=0)
    except ValueError as e:
        # This can happen if batches have inconsistent feature numbers, though less likely if __getitem__ is solid
        warnings.warn(f"Error concatenating feature batches for scaler: {e}. Trying to stack individual items if possible.")
        # Fallback to individual item processing if concatenation fails (slower)
        all_features_list_individual = []
        for feature_item, _ in tqdm(dataset_to_scale, desc="Fallback: Fitting Scaler item by item"):
            if isinstance(feature_item, Tensor):
                feature_item = feature_item.cpu().numpy()
            all_features_list_individual.append(feature_item)
        if not all_features_list_individual:
            warnings.warn("No features extracted in fallback. Scaler cannot be fitted.")
            return None
        try:
            all_features_np = np.array(all_features_list_individual)
            if all_features_np.ndim == 1: # e.g. list of 1D arrays of features
                 all_features_np = np.stack(all_features_list_individual)
        except Exception as stack_e:
            warnings.warn(f"Error stacking individual features for scaler: {stack_e}. Scaler cannot be fitted.")
            return None


    if all_features_np.size == 0:
        warnings.warn("Resulting feature array for scaler is empty.")
        return None
        
    # Ensure it's 2D for StandardScaler
    if all_features_np.ndim == 1:
        # This case should ideally not happen if __getitem__ returns (num_features,)
        # and DataLoader batches correctly. But as a safeguard:
        if len(all_features_np) > 0 and isinstance(all_features_np[0], (np.ndarray, list)):
             all_features_np = np.array(all_features_np.tolist()) # Try to force stack
        else: # Cannot make it 2D
            warnings.warn(f"Cannot reshape features of shape {all_features_np.shape} to 2D for StandardScaler.")
            return None


    # Example: if features from __getitem__ were (9,) and now all_features_np is (N, 9)
    # If features from __getitem__ were (1, 128, 200) for spectrograms,
    # all_features_np would be (N, 1, 128, 200). You'd need to reshape this before fitting scaler.
    # This function is best suited for features that are already vector-like (e.g., CE/KL).
    # For CE/KL, assuming all_features_np is (N, 9)
    if all_features_np.ndim != 2:
        warnings.warn(f"Features for scaler are not 2D (shape: {all_features_np.shape}). StandardScaler might fail or produce unexpected results.")
        # You might attempt a reshape if you know the expected feature dimension, e.g.
        # if all_features_np.ndim > 2 and all_features_np.shape[-1] == 9: # Assuming last dim is feature dim
        #    all_features_np = all_features_np.reshape(-1, 9)
        # else:
        #    return None
        # For now, let's assume it should be 2D for CE/KL
        if not (all_features_np.ndim > 1 and all_features_np.shape[1] > 0) : # Basic check if it can be a valid 2D array
            return None
    breakpoint()

    scaler = StandardScaler()
    try:
        scaler.fit(all_features_np)
    except ValueError as e:
        warnings.warn(f"StandardScaler fit failed: {e}. Ensure features are 2D (samples, features). Shape was: {all_features_np.shape}")
        return None
        
    return scaler