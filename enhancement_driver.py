# run enhancement on selected baseline models with noisy speech
# evaluate the enhancement quality with DNSMOS
# take mean of the scores of all models
# question: what is the difference between dnsmos_ovr, dnsmos_sig and dnsmos_background
# - these scores are directly predicted by a trained models, and the training objective is to fit human evaluations
# meaning of scores of dnsmos:
# - dnsmos_sig: 1-5 scale, 5 means speech signal not distorted and 1 means speech signal heavily distorted
# - dnsmos_background: 1-5 scale, 5 means background noise not audible and 1 means background noise very audible
# - dnsmos_ovr: 1-5 scale, 5 means the overall quality is excellent and 1 means the overall quality is terrible
# in hearing experiment, these three scores are independently evaluated, and does not cunduct further calculation
# after get the mean of all model scores, since these three scores are independent, clusters the scores into three groups
# these three group means enahncement process were success, fail and uncertain
# then, acquire the features of noisy speech as x; the mean of the three scores as y
# then, train a classifier to classify the enhancement process into three groups


# TODO
# compare results within stationary and non-stationary noise
# 1. only use dnsmos_ovr score
# 2. use dnsmos_ovr, dnsmos_sig and dnsmos_background scores
# 3. directly use threshold for separating success, fail and uncertain
# 4. use clustering result for separating success, fail and uncertain (but how to determine clusters of success, fail and uncertain)


import torch
import numpy as np
import os
import librosa
import torchaudio
from torchmetrics.audio import DeepNoiseSuppressionMeanOpinionScore
import subprocess
import glob
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
import warnings
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.ce_kl_feature_extraction import NoiseDataPreprocessor
from sklearn.manifold import Isomap
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score



class SpeechEnhancementPipeline:
    def __init__(self, noisy_speech_dir, device, enhanced_dir):
        self.model_ckpt_path = {
            "SGMSE": "./sgmse/ckpts/train_wsj0_2cta4cov_epoch=159.ckpt",
            "SGMSE_BBED": "./SGMSE_BBED/ckpts/epoch=222-pesq=3.04.ckpt",
            "StoRM": "./StoRM/ckpts/storm_wsj0+wind_epoch=72-pesq=0.00.ckpt",
            "CDiffuSE": "CDiffuSE.ckpt"
        }
        self.noisy_speech_dir = noisy_speech_dir
        self.enhanced_dir = enhanced_dir
        self.target_sr = 16000
        self.device = device
        self.cdiffuse_id = '370200'
    
    def run_enhancement(self, model_name):
        breakpoint()
        # Run the enhancement process on the dataset /sgmse/enhancement.py
        if model_name == "SGMSE":
            enhanced_dir = os.path.join(self.enhanced_dir,model_name)
            breakpoint()
            if os.path.exists(enhanced_dir) and os.path.isdir(enhanced_dir):
                print(f"{enhanced_dir} already exists, skipping enhancement.")
                return f"Enhancement by {model_name} already done."
            # python enhancement.py --test_dir <your_test_dir> --enhanced_dir <your_enhanced_dir> --ckpt <path_to_model_checkpoint>
            subprocess.run(["python",f"/success_fail_estimation/{model_name.lower()}/enhancement.py", 
                            "--test_dir", self.noisy_speech_dir,
                            "--enhanced_dir", os.path.join(self.enhanced_dir,model_name), 
                            "--ckpt", self.model_ckpt_path[model_name]])
            pass
        elif model_name == "SGMSE_BBED":
            # python eval.py --test_dir <your_test_dir> --folder_destination <your_enhanced_dir> --ckpt <path_to_model_checkpoint> --N 30 --reverse_starting_point 0.5 --force_N 15
            subprocess.run(["python",f"/success_fail_estimation/{model_name}/eval.py", 
                            "--test_dir", self.noisy_speech_dir, 
                            "--destination_folder", os.path.join(self.enhanced_dir,model_name), 
                            "--ckpt", self.model_ckpt_path[model_name], 
                            "--N", "30", "--reverse_starting_point", "0.5", "--force_N", "15"])
            pass
        elif model_name == "StoRM":
            enhanced_dir = os.path.join(self.enhanced_dir,model_name)
            breakpoint()
            if os.path.exists(enhanced_dir) and os.path.isdir(enhanced_dir):
                print(f"{enhanced_dir} already exists, skipping enhancement.")
                return f"Enhancement by {model_name} already done."
            # python enhancement.py --test_dir <your_test_dir> --enhanced_dir <your_enhanced_dir> --ckpt <path_to_model_checkpoint>
            subprocess.run(["python",
                            f'/success_fail_estimation/{model_name}/enhancement.py', 
                            "--test_dir", self.noisy_speech_dir, 
                            "--enhanced_dir", os.path.join(self.enhanced_dir,model_name), 
                            "--ckpt", self.model_ckpt_path[model_name], "--mode","storm"])
            pass
        elif model_name == "CDiffuSE":
            enhanced_dir = os.path.join(self.enhanced_dir,model_name)
            if os.path.exists(enhanced_dir) and os.path.isdir(enhanced_dir):
                print(f"{enhanced_dir} already exists, skipping enhancement.")
                return f"Enhancement by {model_name} already done."
            # ./inference.sh [stage] [model name] [checkpoint id]
            # should run different script based on dataset since path is different
            subprocess.run(["./CDiffuSE/inference_dns.sh", "0", model_name, self.cdiffuse_id])
            pass
        pass

class QualityEvaluation:
    def __init__(self, model, enhanced_dir, device,record_dir):
        self.model = model
        self.enhanced_dir = enhanced_dir
        self.device = device
        self.target_sr = 16000
        self.dnsmos_model = DeepNoiseSuppressionMeanOpinionScore(fs=self.target_sr, personalized=False, device=self.device)
        self.dnsmos_scores = {}
        self.record_dir = record_dir

    def evaluate(self):
        # Evaluate the model on the dataset
        # Load the enhanced audio files
        breakpoint()
        if not os.path.exists(os.path.join(self.enhanced_dir)): #taken self.model
            print(f"{os.path.join(self.enhanced_dir,self.model)} does not exist, skipping evaluation.")
            return None
        if not os.path.isdir(os.path.join(self.enhanced_dir)): #taken self.model
            print(f"{os.path.join(self.enhanced_dir,self.model)} is not a directory, skipping evaluation.")
            return None
        # if os.path.exists(os.path.join(self.record_dir, self.model)):
        #     print(f"{os.path.join(self.record_dir, self.model)} already exists, skipping evaluation.")
        #     return None
        breakpoint()
        enhanced_files = os.listdir(self.enhanced_dir) # taken self.model
        for enhanced_file in enhanced_files:
            if enhanced_file.endswith('.wav'):
                file_name = os.path.basename(enhanced_file)
                # Load the enhanced audio file
                y, sr = torchaudio.load(os.path.join(self.enhanced_dir, enhanced_file)) #taken self.model
                # Calculate the DNSMOS score
                dnsmos_score = self.dnsmos_model(y)
                if file_name not in self.dnsmos_scores:
                    self.dnsmos_scores[file_name] = []
                self.dnsmos_scores[file_name] = dnsmos_score.tolist()
        breakpoint()
        # Calculate the mean of the scores across all models
        # for file_name, scores in self.dnsmos_scores.items():
        #     mean_score = np.mean(np.array(scores), axis=0)
        #     self.dnsmos_scores[file_name] = mean_score

        self.labeling(method='threshold')
        # Save the scores to a CSV file
        df = pd.DataFrame(self.dnsmos_scores).T
        df.columns = ['dnsmos_ovr1','dnsmos_sig', 'dnsmos_background', 'dnsmos_ovr', 'label']
        breakpoint()
        if not os.path.exists(os.path.join(self.record_dir, self.model)):
            os.makedirs(os.path.join(self.record_dir, self.model))
        df.to_csv(os.path.join(self.record_dir, self.model, 'dnsmos_scores.csv'), index=True)  
  
    def labeling(self, method = 'threshold'):
        # Label the enhancement process as success, fail, or uncertain
        # based on the mean of the three score
        if method == 'threshold':
            fail_threshold = 2
            success_threshold = 2.75
            for file_name, scores in self.dnsmos_scores.items():
                if scores[-1] < fail_threshold:
                    label = -1 # fail
                elif scores[-1] > success_threshold:
                    label = 1 # success
                else:
                    label = 0 # uncertain
                self.dnsmos_scores[file_name].append(label)
            # Save the labels to a CSV file
            # df = pd.DataFrame(self.dnsmos_scores)
        elif method == 'clustering':
            cluster_model = DnsmosClustering(self.record_dir)
            cluster_model.load_dnsmos_scores()
            cluster_model.cluster()
            cluster_model.dimension_reduction()
            cluster_model.visualize_clustering()
            labels = cluster_model.evaluate()
            for file_name, scores in self.dnsmos_scores.items():
                if file_name in labels:
                    self.dnsmos_scores[file_name] = labels[file_name]
                else:
                    self.dnsmos_scores[file_name] = 'uncertain'
            


        pass



class Classifier:
    def __init__(self):
        self.decision_tree = DecisionTreeClassifier()
        self.svm = SVC(kernel='linear')
        self.random_forest = None

    def train(self, features, labels):
        # Train the classifier
        self.decision_tree.fit(features, labels)
        self.svm.fit(features, labels)
        # self.random_forest.fit(features, labels)

    def predict(self, features):
        self.decision_tree.predict(features)
        self.svm.predict(features)
        # self.random_forest.predict(features)
        pass

    def predict_prob(self, features):
        # Predict the probability of success or failure
        self.decision_tree.predict_proba(features)
        self.svm.predict_proba(features)
        # self.random_forest.predict_proba(features)
        pass

class Regressor:
    '''
    to estimate how likely a sample is to be success or fail across models
    - g.e. 3: all models predict success
    - 2: two models predict success and one model predict uncertain;
    - 1: one model predict success and two models predict uncertain; two model predict success and one model predict fail;
    - 0: all models predict uncertain;
    - -1: one model predict fail and two models predict uncertain; two model predict fail and one model predict success;
    - -2: two models predict fail and one model predict uncertain; 
    - -3: all models predict fail;
    '''
    def __init__(self):
        self.decision_tree = DecisionTreeRegressor()
        self.svm = SVR(kernel='rbf')
        self.random_forest = None

    def train(self, features, labels):
        # Train the regressor
        self.decision_tree.fit(features, labels)
        self.svm.fit(features, labels)
        # self.random_forest.fit(features, labels)
        pass

    def predict(self, features):
        # Predict the success or failure of the enhancement process
        self.decision_tree.predict(features)
        self.svm.predict(features)
        # self.random_forest.predict(features)
        pass

class DnsmosClustering:
    def __init__(self, dnsmos_scores_dir):
        self.kmeans = KMeans(n_clusters=3)
        self.dnsmos_scores_dir = dnsmos_scores_dir
        self.fname_id_mapping = {}

    def load_dnsmos_scores(self):
        breakpoint()
        # check name of subdirectories same as model name, and scores file in subdir is dnsmos_scores.csv
        model_names = os.listdir(self.dnsmos_scores_dir)
        dataset_path = []
        for model_name in model_names:
            if os.path.isdir(os.path.join(self.dnsmos_scores_dir, model_name)):
                # check if dnsmos_scores.csv exists in the subdirectory
                if os.path.exists(os.path.join(self.dnsmos_scores_dir, model_name, 'dnsmos_scores.csv')):
                    # load the dnsmos_scores.csv file
                    dataset_path.append(os.path.join(self.dnsmos_scores_dir, model_name, 'dnsmos_scores.csv'))
                    
        
        datasets = []
        # Load the DNSMOS scores from the CSV file
        for dataset in dataset_path:
            if os.path.exists(dataset):
                df = pd.read_csv(dataset)
                datasets.append(df)
        breakpoint()

        # Concatenate the datasets and calculate the mean of the scores by row
        if len(datasets) > 1:
            self.dnsmos_scores = pd.concat(datasets, axis=0)
            breakpoint()
            self.fname_id_mapping = self.dnsmos_scores['Unnamed: 0'].to_dict()
            self.dnsmos_scores = self.dnsmos_scores.groupby(self.dnsmos_scores['Unnamed: 0'])[['dnsmos_ovr1','dnsmos_sig', 'dnsmos_background', 'dnsmos_ovr']].mean() # for label, may be use sum instead of mean? (indicate the sample can be well handled by all used model or not)
            self.labels = self.dnsmos_scores.groupby(self.dnsmos_scores['Unnamed: 0'])['label'].sum()
            self.dnsmos_scores = self.dnsmos_scores.reset_index(drop=True)
            self.dnsmos_scores = self.dnsmos_scores[['dnsmos_ovr1','dnsmos_sig', 'dnsmos_background', 'dnsmos_ovr']].values
        elif len(datasets) == 1:
            self.dnsmos_scores = datasets[0][['dnsmos_ovr1','dnsmos_sig', 'dnsmos_background', 'dnsmos_ovr']].values

        pass

    def dimension_reduction(self):
        # Perform dimension reduction on the dnsmos scores
        isomap = Isomap(n_components=2)
        self.dnsmos_scores_reduced = isomap.fit_transform(self.dnsmos_scores)
        pass

    def visualize_clustering(self):
        # Visualize the clustering result

        plt.figure(figsize=(10, 6))
        plt.scatter(self.dnsmos_scores_reduced[:, 0], self.dnsmos_scores_reduced[:, 1], c=self.kmeans.labels_, cmap='viridis', marker='o')
        plt.title('DNSMOS Clustering')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.colorbar(label='Cluster Label')
        plt.savefig(os.path.join(self.dnsmos_scores_dir, 'dnsmos_clustering.png'))
        plt.show()

        pass

    def cluster(self):
        # Cluster the dnsmos scores into three groups
        self.kmeans.fit(self.dnsmos_scores)
        pass

    def predict(self):
        # Predict the enhancement process based on the clustered dnsmos scores
        labels = self.kmeans.predict(self.dnsmos_scores)
        # Map the labels to the original file names
        self.fname_success_fail = {self.fname_id_mapping[i]: labels[i] for i in range(len(labels))}
        # evaluate the clustering result


        pass

    def evaluate(self, stage):
        # Evaluate the clustering result
        if stage == 'train':
            sil_score = silhouette_score(self.dnsmos_scores, self.kmeans.labels_)
            print(f"Silhouette Score: {sil_score}")
        elif stage == 'test':
            # measure the similarity between the clustering result and the original labels
            # load the original labels
            
            original_labels = pd.read_csv(os.path.join(self.dnsmos_scores_dir, 'original_labels.csv'))
            original_labels = original_labels.set_index('Unnamed: 0')
            original_labels = original_labels['label'].to_dict()
            # map the labels to the original file names
            self.fname_id_mapping = {self.fname_id_mapping[i]: original_labels[i] for i in range(len(original_labels))}
            # calculate the similarity
            # calculate the adjusted rand index
            ari = adjusted_rand_score(list(original_labels.values()), list(self.fname_id_mapping.values()))
            # calculate the normalized mutual information score
            nmi = normalized_mutual_info_score(list(original_labels.values()), list(self.fname_id_mapping.values()))
            # calculate the homogeneity score
            homogeneity = homogeneity_score(list(original_labels.values()), list(self.fname_id_mapping.values()))
            # calculate the completeness score
            completeness = completeness_score(list(original_labels.values()), list(self.fname_id_mapping.values()))
            # calculate the v measure score
            v_measure = v_measure_score(list(original_labels.values()), list(self.fname_id_mapping.values()))

            # print the scores
            print(f"Adjusted Rand Index: {ari}")
            print(f"Normalized Mutual Information Score: {nmi}")
            print(f"Homogeneity Score: {homogeneity}")
            print(f"Completeness Score: {completeness}")
            print(f"V Measure Score: {v_measure}")

        pass


class SuccessFailEstimation:
    def __init__(self, data_loader, device):
        self.data_loader = data_loader
        self.device = device

    def estimate(self):
        # Estimate the success or failure of the enhancement process
        pass


if __name__ == "__main__":
    # Define the directories and device
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--noisy_speech_dir", type=str, required=True, help="Path to the noisy speech directory")
    arg_parser.add_argument("--dataset_name", type=str, required=True, help="Path to the target directory")
    arg_parser.add_argument("--model_name", type=str, required=True, help="Name of the model to be used")
    arg_parser.add_argument("--record_dir", type=str, required=True, help="Path to the record directory")
    arg_parser.add_argument('--labeling_method', type=str, default='threshold', choices=['threshold', 'clustering'], help='Method for labeling the enhancement process')
    arg_parser.add_argument('--enhanced_dir', type=str, default=None, help='Path to the enhanced directory, if not provided, will be created based on model name')
    args = arg_parser.parse_args()

    noisy_speech_dir = args.noisy_speech_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    breakpoint()
    # Initialize the pipeline
    pipeline = SpeechEnhancementPipeline(args.noisy_speech_dir, device, args.enhanced_dir)

    # # Run the enhancement process on the selected models
    pipeline.run_enhancement(args.model_name)
    # Initialize the quality evaluation
    enhanced_dir = os.path.join(args.enhanced_dir, args.model_name)
    if args.model_name == 'CDiffuSE':
        enhanced_dir = '/success_fail_estimation/ljspeech_esc50/enhanced/Enhanced/CDiffuSE/model370200/test/spec'
    quality_eval = QualityEvaluation(args.model_name, enhanced_dir, device, args.record_dir)
    
    # Evaluate the models
    dnsmos_scores = quality_eval.evaluate()
    # Label the enhancement process
    quality_eval.labeling(method=args.labeling_method)

    # breakpoint()
    # # DNSMOS clustering and visualization
    # dnsmos_clustering = DnsmosClustering(args.record_dir)
    # dnsmos_clustering.load_dnsmos_scores()
    # dnsmos_clustering.cluster()
    # dnsmos_clustering.dimension_reduction()
    # dnsmos_clustering.visualize_clustering()
    # dnsmos_clustering.evaluate(stage='train')
