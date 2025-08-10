import pandas as pd
from torchmetrics.audio import DeepNoiseSuppressionMeanOpinionScore
from pathlib import Path
import torchaudio
from utils.ce_kl_feature_extraction import NoiseDataPreprocessor
from scipy.stats import ttest_ind


class HardClassifyCasesAnalysis:
    def __init__(self):
        self.dns_mos = DeepNoiseSuppressionMeanOpinionScore(
            fs=16000,  # Assuming a sample rate of 16kHz
            personalized=False,
        )
        self.noise_data_preprocessor = NoiseDataPreprocessor(audio_paths=None)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000)
        self.results_dir = './'

    def analyze(self):
        # Load the hard classify cases
        hard_cases_path = "/success_fail_estimation/expert_correctly_classified_samples_fold1.csv"
        hard_cases_df = pd.read_csv(hard_cases_path)

        # Initialize a list to store results
        results = []
        models = ['SGMSE','StoRM','CDiffuSE']

        for _, row in hard_cases_df.iterrows():
            all_dnsmos = {}
            all_features = {}
            for model_name in models:
                file_path = row['file_path']
                dataset_name = file_path.split('/')[2]
                if dataset_name == 'voicebank':
                    dataset_name = dataset_name + '/noisy_testset_wav'
                elif dataset_name == 'crema_mssnsd_non_stationary':
                    dataset_name = dataset_name + '/test/noisy'
                fname = file_path.split('/')[-1]
                audio_path = f"/{file_path.split('/')[1]}/{dataset_name}/enhanced/{model_name}/{fname}"

                if model_name == 'CDiffuSE':
                    if dataset_name == 'crema_mssnsd_non_stationary/test/noisy':
                        audio_path = f"/{file_path.split('/')[1]}/{dataset_name}/enhanced/{model_name}/{fname}"
                    else:
                        audio_path = f"/{file_path.split('/')[1]}/{dataset_name}/enhanced/Enhanced/CDiffuSE/model370200/test/spec/{fname}"
                # load corresponding audio file in different datasets
                if not Path(audio_path).exists():
                    continue
                audio, sr = torchaudio.load(audio_path)
                # Compute MOS for the audio file
                mos_score = self.dns_mos(audio)
                all_dnsmos[model_name] = mos_score.tolist()[-1]

            ori_audio, sr = torchaudio.load(row['file_path'])
            ori_audio_melspec = self.mel_transform(ori_audio)
            ce_kl_features = self.noise_data_preprocessor.extract_features_by_sample(
                ori_audio_melspec.squeeze(0).numpy(), sr
            )
            results.append({
                'file_path': row['file_path'],
                'SGMSE_MOS': all_dnsmos.get('SGMSE', None),
                'StoRM_MOS': all_dnsmos.get('StoRM', None),
                'CDiffuSE_MOS': all_dnsmos.get('CDiffuSE', None),
                # 'true_label': row['true_label'],
                # 'predicted_label': row['predicted_label'],
                # 'variance_of_ce': ce_kl_features[0],
                # 'mean_off_diagonal_ce': ce_kl_features[1],
                # 'variance_off_diagonal_ce': ce_kl_features[2],
                # 'mean_adjacent_ce': ce_kl_features[3],
                # 'variance_adjacent_ce': ce_kl_features[4],
                # 'mean_off_diagonal_kl': ce_kl_features[5],
                # 'variance_off_diagonal_kl': ce_kl_features[6],
                # 'mean_adjacent_kl': ce_kl_features[7],
                # 'variance_adjacent_kl': ce_kl_features[8],
            })
        # Convert results to DataFrame and save
        results_df = pd.DataFrame(results)
        print(results_df.describe())
        # calculate the mean and variance difference of bset MOS scores and second leading MOS scores
        breakpoint()
        results_df['best_mos'] = results_df[['SGMSE_MOS', 'StoRM_MOS', 'CDiffuSE_MOS']].max(axis=1)
        results_df['second_best_mos'] = results_df[['SGMSE_MOS', 'StoRM_MOS', 'CDiffuSE_MOS']].apply(
            lambda x: sorted(x, reverse=True)[1], axis=1
        )
        results_df['mos_diff'] = results_df['best_mos'] - results_df['second_best_mos']
        results_df.to_csv("./correctly_classify_cases_analysis.csv", index=False)
        results_df.describe().to_csv("./correctly_classify_cases_analysis_summary.csv")


class CasesAnalysis:
    def __init__(self, expert_case_path, low_confidence_case_path):
        self.expert_case_lists = pd.read_csv(expert_case_path)
        self.low_confidence_case_lists = pd.read_csv(low_confidence_case_path)


    def analyze(self):
        # Load the low confidence cases
        expert_cols = self.expert_case_lists[[#'true_label',
                         #'gatekeeper_pred_label',
                         # 'expert_confidence',
                         #'uncertainty_score',
                         'diagonal variance of CE',
                         'mean off-diagonal of CE',
                         'variance off-diagonal of CE',
                         'mean adjacent frame of CE',
                         'variance adjacent frame of CE',
                         'mean off-diagonal of KL divergence',
                         'variance off-diagonal of KL divergence',
                         'mean adjacent frame of KL divergence',
                         'variance adjacent frame of KL divergence']]
        low_confidence_cols = self.low_confidence_case_lists[[#'true_label',
                         #'gatekeeper_pred_label',
                         # 'expert_confidence',
                         #'uncertainty_score',
                         'variance_of_ce',
                         'mean_off_diagonal_ce',
                         'variance_off_diagonal_ce',
                         'mean_adjacent_ce',
                         'variance_adjacent_ce',
                         'mean_off_diagonal_kl',
                         'variance_off_diagonal_kl',
                         'mean_adjacent_kl',
                         'variance_adjacent_kl']]
        
        ttest_result = {}
        for i,col in enumerate(expert_cols.columns):
            expert_col = expert_cols[col]
            low_col = low_confidence_cols.columns[i]
            low_confidence_col = low_confidence_cols[low_col]
            t_statistic, p_value = ttest_ind(expert_col, low_confidence_col, equal_var=False)
            print(f"T-test for {col}: t-statistic = {t_statistic}, p-value = {p_value}")
            ttest_result[col] = {
                't_statistic': t_statistic,
                'p_value': p_value
            }
        breakpoint()
        # Create a DataFrame to summarize the results
        describe = pd.DataFrame({
            'expert_mean': expert_cols.mean(),
            'expert_std': expert_cols.std(),
            'low_confidence_mean': low_confidence_cols.mean(),
            'low_confidence_std': low_confidence_cols.std(),
            't_statistic': [ttest_result[col]['t_statistic'] for col in expert_cols.columns],
            'p_value': [ttest_result[col]['p_value'] for col in expert_cols.columns]
        })
        print(describe)
        # Save the description to a CSV file
        describe.to_csv("./expert_cases_analysis_summary.csv")


if __name__ == "__main__":
    analysis = HardClassifyCasesAnalysis()
    analysis.analyze()
    # print("Hard classify cases analysis completed and saved to hard_classify_cases_analysis.csv")
    # expert_case_path = '/success_fail_estimation/expert_correctly_classified_samples_fold1.csv'
    # low_confidence_case_path = './hard_classify_cases_analysis.csv'
    # analysis = CasesAnalysis(expert_case_path, low_confidence_case_path)
    # analysis.analyze()