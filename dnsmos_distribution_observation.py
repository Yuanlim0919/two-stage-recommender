# read all dnsmos records generated from all enhancement models
# compare direction 1: for a same dataset, how the consistency of success/fail/uncertain label looks like (Cross entropy)
# compare direction 2: for a same model, how the consistency of success/fail/uncertain label looks like ()

import math
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
# cross entropy
from sklearn.metrics import log_loss
from argparse import ArgumentParser
from scipy.stats import entropy
from utils.ce_kl_feature_extraction import NoiseDataPreprocessor
import torchaudio

class DNMSOSDistributionObservation:
    def __init__(self):
        self.dnsmos_dir = [
            #'/success_fail_estimation/dnsmos_records_crema_mssnsd',
            # '/success_fail_estimation/dnsmos_records_voicebank',
            # '/success_fail_estimation/dnsmos_record_dns'
            # '/success_fail_estimation/dnsmos_records_voicebank_train',
            '/success_fail_estimation/dnsmos_records_ljspeech_esc50'
        ]
        self.speech_dirs = ['/success_fail_estimation/sgmse/noisy_trainset_28spk_wav']
        self.dnsmos_df = None

    def cross_entropy(self,p_dist, q_dist, base=2, epsilon=1e-12):
        """
        Calculates the cross-entropy H(P, Q).
        H(P,Q) = - sum_x P(x) * log_base(Q(x))

        Args:
            p_dist (dict): Probability distribution P. {label: prob}
            q_dist (dict): Probability distribution Q. {label: prob}
            base (int): The base of the logarithm (e.g., 2 for bits, math.e for nats).
            epsilon (float): Small value to add to Q's probabilities to avoid log(0).
                            Set to 0 to get 'inf' if Q(x)=0 for P(x)>0.
        Returns:
            float: The cross-entropy. Can be float('inf').
        """
        breakpoint()
        ce = 0.0
        for label, p_x in p_dist.items():
            if p_x > 0: # Only consider terms where P(x) > 0
                q_x = q_dist.get(label, 0.0) # Get Q(x), default to 0 if label not in Q
                
                if q_x == 0.0:
                    if epsilon > 0: # If we allow epsilon smoothing for Q
                        q_x_smooth = epsilon 
                    else: # Strict calculation, Q(x)=0 -> log(Q(x)) = -inf -> CE = inf
                        return float('inf') 
                else:
                    q_x_smooth = q_x

                if q_x_smooth <= 0: # Should not happen if epsilon > 0 and q_x = 0
                    return float('inf') # Safety for log of non-positive

                ce -= p_x * math.log(q_x_smooth, base)
        return ce

    def kl_divergence(self, p_dist, q_dist, base=2, epsilon=1e-12):
        """
        Calculates KL Divergence D_KL(P || Q)
        D_KL(P || Q) = sum_x P(x) * log_base(P(x) / Q(x))
                    = H(P,Q) - H(P)
        """
        ce_pq = self.cross_entropy(p_dist, q_dist, base, epsilon)
        if ce_pq == float('inf'):
            return float('inf')
        
        entropy_p = 0.0
        for p_x in p_dist.values():
            if p_x > 0:
                entropy_p -= p_x * math.log(p_x, base)
                
        return ce_pq - entropy_p

    def load_dnsmos_data(self):
        # collect all the label predicted, and for each model, calculate cross entropy of predicted label across dataset
        model_names = ['CDiffuSE','SGMSE','StoRM']
        self.dnsmos_dfs = {}
        # pair up the label predicted across dataset by model
        # for each model, calculate the cross entropy of predicted label across dataset
        model_label_dict = {}
        for model in model_names:
            labels = []
            # identify the dnsmos_scores.csv in the model directory
            for dir in self.dnsmos_dir:
                if os.path.basename(dir) == 'dnsmos_records_ljspeech_esc50':
                    model_dir = os.path.join(dir, model)
                    if os.path.exists(model_dir):
                        # read the dnsmos_scores.csv file
                        dnsmos_scores = pd.read_csv(os.path.join(model_dir, 'dnsmos_scores.csv'))
                        self.dnsmos_dfs[f'{model}'] = dnsmos_scores
                        # check if the dataframe has a column named 'label'
                        if 'label' in dnsmos_scores.columns:
                            labels.append(dnsmos_scores['label'].values.tolist())
            # add to the dictionary
            model_label_dict[model] = labels
        self.distribution_by_model = {}
        # calculate the similarity of the label predicted across dataset by model
        for model, all_labels in model_label_dict.items():
            distribution = []
            # calculate probability distribution of each label
            for label in all_labels:
                # calculate the probability distribution of the label
                label_distribution = {}
                for i in range(len(label)):
                    if label[i] not in label_distribution:
                        label_distribution[label[i]] = 1
                    else:
                        label_distribution[label[i]] += 1
                # normalize the distribution
                for key in label_distribution.keys():
                    label_distribution[key] /= len(label)
                # add to the model_label_dict
                distribution.append(label_distribution)
            self.distribution_by_model[model] = distribution

    def dnsmos_distribution_by_models(self):

        # calculate the cross entropy of the label distribution across dataset by model
        cross_entropy = {}
        for model, distribution in self.distribution_by_model.items():
            cross_entropy[model] = []
            for i in range(len(distribution)):
                for j in range(i + 1, len(distribution)):
                    # calculate the cross entropy
                    cross_entropy[model].append(self.cross_entropy(distribution[i], distribution[j]))
        print(cross_entropy)

        # plot distribution by model as barchart
        fig, ax = plt.subplots(3, 1, figsize=(10, 15))
        for i, (model, distribution) in enumerate(self.distribution_by_model.items()):
            for elems in distribution:
                # plot the distribution
                ax[i].bar(list(elems.keys()), list(elems.values()))
            ax[i].set_title(f'DNMSOS Distribution by Model: {model}')
            ax[i].set_xlabel('Label')
            ax[i].set_ylabel('Probability')
            ax[i].set_xticks(list(elems.keys()))
        # plt.xlabel('Dataset')
        # plt.ylabel('Probability')
        # plt.title('DNMSOS Distribution by Model')
        # plt.xticks(x, [f'Dataset {i + 1}' for i in range(len(distribution))])
        # plt.legend()
        plt.savefig('dnsmos_distribution_by_model.png')

        pass
    
    def dnsmos_distribution_by_dataset(self):

        # calculate the cross entropy for each dataset
        cross_entropy = {}
        dataset_dict = {}
        dataset_name = ['crema_mssnsd', 'voicebank']
        for i in range(len(dataset_name)):
            dataset_dict[dataset_name[i]] = []
            for model, distribution in self.distribution_by_model.items():
                # calculate the cross entropy
                dataset_dict[dataset_name[i]].append(distribution[i])
        # calculate the cross entropy
        for model, distribution in dataset_dict.items():
            cross_entropy[model] = []
            for i in range(len(distribution)):
                for j in range(i + 1, len(distribution)):
                    # calculate the cross entropy
                    cross_entropy[model].append(self.cross_entropy(distribution[i], distribution[j]))
        print(cross_entropy)
    
    def best_model_counting(self,dnsmos_df):
        best_models = []
        dnsmos_df.reset_index(drop=True, inplace=True)
        for i in range(len(dnsmos_df)):
            # check if all models failed
            if all(dnsmos_df.iloc[i][f'{model}_dnsmos_ovr'] <= 2 for model in self.dnsmos_dfs.keys()):
                best_models.append('All Failed')
            else:
                # get the model with the highest dnsmos_ovr
                best_model = dnsmos_df.iloc[i].values[1:].argmax()
                # get the model name
                best_model = list(dnsmos_df.columns)[best_model+1]
                best_models.append(best_model.split('_')[0])
        # count values of best_models
        best_model_counts = {}
        for model in best_models:
            if model not in best_model_counts:
                best_model_counts[model] = 1
            else:
                best_model_counts[model] += 1
        return best_model_counts, best_models

    def get_cekl_features(self, best_model_labels):
        speech_files = [os.path.join(self.speech_dirs[0],fpath) for fpath in os.listdir(self.speech_dirs[0]) if fpath.endswith('.wav')]
        data_preprocessor = NoiseDataPreprocessor(self.speech_dirs[0])
        stft_transform = torchaudio.transforms.Spectrogram(n_fft=400, hop_length=160, power=2.0)
        all_features = []
        for file in speech_files:
            audio, sr = torchaudio.load(file)
            s_power_torch = stft_transform(audio)
            # Squeeze channel dim and convert to NumPy for NoiseDataPreprocessor
            s_power_numpy = s_power_torch.squeeze(0).numpy() # Shape: (freq_bins, time_frames)
            # 2. Pass the power spectrogram to user's feature extraction method
            # The `sr` argument is passed along as the user's method expects it.
            features_numpy = data_preprocessor.extract_features_by_sample(s_power_numpy,sr)
            all_features.append(features_numpy)
        # all_features.append(best_model_labels)
        # Convert list of features to a DataFrame
        features_df = pd.DataFrame(all_features)
        features_df['best_model'] = best_model_labels
        breakpoint()
        return features_df

    def calculate_mean_std_by_models(self, features_df):
        models = features_df['best_model'].unique()
        mean_std_dict = {}
        for model in models:
            model_features = features_df[features_df['best_model'] == model].iloc[:, :-1]
            mean_std_dict[model] = {
                'mean': model_features.mean(),
                'std': model_features.std()
            }
        print(mean_std_dict)
        pass


    def dnsmos_distribution_by_sample(self):
        # left join all the dnsmos_scores.csv files by fname
        # observe the consistency of the labels (may use cross entropy)
        # for each sample, collect the dnsmos labels from all models, and observe the consistency of the labels
        # identify the best performed model by the dnsmos label (how if two model gained a same label?), or the sample will fail in all models

        labels_df = pd.DataFrame()
        dnsmos_df = pd.DataFrame()
        breakpoint()
        for i, (model, df) in enumerate(self.dnsmos_dfs.items()):
            # rename the column 'label' to 'modelname_label'
            df.rename(columns={'label': f'{model}_label','dnsmos_ovr':f'{model}_dnsmos_ovr'}, inplace=True)
            if i == 0:
                labels_df = df[['Unnamed: 0', f'{model}_label']]
                dnsmos_df = df[['Unnamed: 0', f'{model}_dnsmos_ovr']]
            else:
                labels_df = pd.merge(labels_df, df[['Unnamed: 0',f'{model}_label']], on='Unnamed: 0', how='left')
                dnsmos_df = pd.merge(dnsmos_df, df[['Unnamed: 0',f'{model}_dnsmos_ovr']], on='Unnamed: 0', how='left')
        # calculate the cross entropy of the label distribution across dataset by model
        label_cross_entropy = {}
        dnsmos_cross_entropy = {}
        
        # get the values form columns name with 'label'
        label_columns = [labels_df[col].values for col in labels_df.columns if '_label' in col]
        # get the values form columns name with 'dnsmos_ovr'
        dnsmos_columns = [dnsmos_df[col].values for col in dnsmos_df.columns if '_dnsmos_ovr' in col]
        # calculate the cross entropy of the dnsmos distribution across dataset by model
        for i in range(len(label_columns)):
            for j in range(i + 1, len(label_columns)):
                # calculate the cross entropy
                #label_cross_entropy[f'{i}_{j}'] = log_loss(label_columns[i], label_columns[j])
                # -np.sum(pk * np.log(qk)) / np.log(base)
                dnsmos_cross_entropy[f'{i}_{j}'] = entropy(np.round(dnsmos_columns[i]).astype(int),base=2)+entropy(np.round(dnsmos_columns[i]).astype(int), np.round(dnsmos_columns[j]).astype(int),base=2)
        
        # determine the best model for each sample, or all models failed (all model dnsmos_ovr<=2)
        breakpoint()
        best_model_count, best_models = self.best_model_counting(dnsmos_df)
        # cekl_features = self.get_cekl_features(best_models)
        # self.calculate_mean_std_by_models(cekl_features)
        print(best_model_count)
        breakpoint()
        # plot the distribution of dnsmos_ovr by model as violinplot
        fig, ax = plt.subplots(figsize=(10, 8))
        # plot the distribution of dnsmos_ovr by model
        sns.violinplot(data=dnsmos_df.iloc[:, 1:], ax=ax)
        ax.set_title('DNMSOS Distribution by Model')
        ax.set_xlabel('Model')
        ax.set_ylabel('DNMSOS OVR')
        plt.xticks(rotation=30)
        plt.savefig(f'dnsmos_distribution_{os.path.basename(self.dnsmos_dir[0])}.png')

        breakpoint()
        # for bad cases (all dnsmos scores <= 2), analyze the chacteristics of the samples (dns, crema_mssnsd)
        feature_columns_to_check = [
            f'{model}_dnsmos_ovr' for model in self.dnsmos_dfs.keys()
            if f'{model}_dnsmos_ovr' in dnsmos_df.columns # Ensure column exists
        ]
        bad_mask = dnsmos_df[feature_columns_to_check].le(2).all(axis=1)
        bad_cases = dnsmos_df[bad_mask]

        ideal_mask = dnsmos_df[feature_columns_to_check].gt(3).all(axis=1) 
        ideal_cases = dnsmos_df[ideal_mask]   

        # mediocre_mask = ((dnsmos_df[feature_columns_to_check] > 2) & (dnsmos_df[feature_columns_to_check] < 3)).any(axis=1)
        # mediocre mask is all other than bad and ideal cases (may contain some case which one model perform well but others failed)
        mediocre_mask = ~bad_mask & ~ideal_mask
        mediocre_cases = dnsmos_df[mediocre_mask] #242
        breakpoint()
        

        # save the bad cases to a csv file
        bad_cases.to_csv(f'bad_cases_{os.path.basename(self.dnsmos_dir[0])}.csv', index=False)
        ideal_cases.to_csv(f'ideal_cases_{os.path.basename(self.dnsmos_dir[0])}.csv', index=False)
        mediocre_cases.to_csv(f'mediocre_cases_{os.path.basename(self.dnsmos_dir[0])}.csv', index=False)

        print(f'Bad cases: {len(bad_cases)}, Ideal cases: {len(ideal_cases)}, Mediocre cases: {len(mediocre_cases)}')

        # save the file name and corresponding best model into a csv
        dnsmos_df['best_model'] = best_models
        dnsmos_df.to_csv(f'best_model_{os.path.basename(self.dnsmos_dir[0])}.csv', index=False)
        pass


if __name__ == '__main__':
    parser = ArgumentParser(description="DNMSOS Distribution Observation")
    parser.add_argument('--model_or_dataset', type=str, default='model', choices=['model', 'dataset','sample'],
                        help='observe dnsmos distribution by model or dataset')
    args = parser.parse_args()
    dnsmos_observation = DNMSOSDistributionObservation()
    dnsmos_observation.load_dnsmos_data()
    if args.model_or_dataset == 'model':
        dnsmos_observation.dnsmos_distribution_by_models()
    elif args.model_or_dataset == 'dataset':
        dnsmos_observation.dnsmos_distribution_by_dataset()
    elif args.model_or_dataset == 'sample':
        dnsmos_observation.dnsmos_distribution_by_sample()