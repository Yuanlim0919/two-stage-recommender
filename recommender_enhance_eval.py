# Input: noisy speech dir
# Output: enhanced speech dir
# Steps:
# - Load recommender model
# - Load noisy speech files
# - make recommendation
# - based on recommendation result, create subdirs to store noisy speech files
# - run enhancement by recommended model with subprocess
# - evaluate the quality of enhanced speech files with PESQ, STOI and DNSMOS

import os
import subprocess
import timm
import numpy as np
from pathlib import Path
import torchaudio
import torchaudio.transforms as T
import torch
from torch.utils.data import DataLoader, Dataset
from utils.ce_kl_feature_extraction import NoiseDataPreprocessor
from enhancement_driver import SpeechEnhancementPipeline
from model_recommender_w_scaler import RecommenderModelCeKl, RecommenderModelMelSpec, RecommenderModelMelSpecPretrain, RecommenderModelAST, RecommenderModelASTHybrid
from torchmetrics.audio import PerceptualEvaluationSpeechQuality, ShortTimeObjectiveIntelligibility, DeepNoiseSuppressionMeanOpinionScore
from transformers import ASTModel, ASTFeatureExtractor
from tqdm import tqdm


class RecommenderBasedEnhancement:
    def __init__(self, noisy_speech_dir, output_dir, feature_type=None, model_path=None, 
                 gatekeeper_path=None, expert_path=None, use_hybrid_features=False,
                 cekl_features_base_dir=None):
        self.model_path = model_path
        self.noisy_speech_dir = noisy_speech_dir
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_type = feature_type
        self.preprocessor = NoiseDataPreprocessor(audio_paths=noisy_speech_dir)
        self.label_map = {
            0: 'All Failed', 1: 'CDiffuSE', 2: 'SGMSE', 3:'StoRM'
        }
        self.gatekeeper_path = gatekeeper_path
        self.expert_path = expert_path
        self.use_hybrid_features = use_hybrid_features
        self.label_map = {0: 'All Failed', 1: 'CDiffuSE', 2: 'SGMSE', 3:'StoRM'}
        self.expert_prediction_map_inv = {0: 0, 1: 1, 2: 3} 
        self.cekl_features_base_dir = Path(cekl_features_base_dir) if cekl_features_base_dir else None
        if model_path is not None and Path(model_path).exists():
            backbone = timm.create_model(
                model_name=args.pretrained_models, 
                pretrained=True, 
                num_classes=0,
                in_chans=1
            )
            model = RecommenderModelMelSpecPretrain(
                backbone=backbone,
                num_backbone_features=1280
            )
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            self.model = model.eval()

        elif gatekeeper_path and expert_path:
            hf_model_path = 'MIT/ast-finetuned-audioset-10-10-0.4593'
            self.feature_extractor = ASTFeatureExtractor.from_pretrained(hf_model_path)

            # Load Gatekeeper (binary classifier)
            gatekeeper_backbone = ASTModel.from_pretrained(hf_model_path)
            self.gatekeeper_model = RecommenderModelASTHybrid(backbone=gatekeeper_backbone, output_size=2)
            self.gatekeeper_model.load_state_dict(torch.load(self.gatekeeper_path, map_location=self.device))
            self.gatekeeper_model.to(self.device)
            self.gatekeeper_model.eval()

            # Load Expert (3-class classifier)
            expert_backbone = ASTModel.from_pretrained(hf_model_path)
            if self.use_hybrid_features:
                print("Loading HYBRID Expert Model.")
                self.expert_model = RecommenderModelASTHybrid(backbone=expert_backbone, output_size=3)
                if self.cekl_features_base_dir is None:
                    raise ValueError("`cekl_features_base_dir` must be provided when using hybrid expert.")
            else:
                print("Loading STANDARD Expert Model.")
                self.expert_model = RecommenderModelAST(backbone=expert_backbone, output_size=3)
                
            self.expert_model.load_state_dict(torch.load(self.expert_path, map_location=self.device))
            self.expert_model.to(self.device).eval()
        pass

    def get_features(self, audio_path):
        """Prepares all necessary features for a single audio file."""
        waveform, sr = torchaudio.load(audio_path)
        target_sr = self.feature_extractor.sampling_rate

        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)
        
        if waveform.ndim > 1:
            waveform = torch.mean(waveform, dim=0)

        # Get spectrogram features
        inputs = self.feature_extractor(waveform, sampling_rate=target_sr, return_tensors="pt")
        spec_features = inputs['input_values'].to(self.device)

        # Get CE/KL features if needed
        if self.use_hybrid_features:
            # Construct path to pre-calculated .npy file
            relative_path = audio_path.relative_to(Path(self.noisy_speech_dir).parent) # Adjust base if needed
            if str(relative_path.with_suffix('.npy').parent) == 'noisy_testset_wav':
                cekl_path = self.cekl_features_base_dir / 'voicebank' / relative_path.with_suffix('.npy')
            else:
                cekl_path = self.cekl_features_base_dir / relative_path.with_suffix('.npy')
            cekl_numpy = np.load(cekl_path)
            cekl_features = torch.tensor(cekl_numpy, dtype=torch.float32).unsqueeze(0).to(self.device)
            return waveform, sr, spec_features, cekl_features
        else:
            return spec_features, None

    def extract_features(self, audio, sr):
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=16000)
        if self.feature_type == 'mel_spec':
            transform = T.MelSpectrogram(sample_rate=16000,
                 n_mels=128, n_fft=400, hop_length=160)
            features = transform(audio)
            features = features.unsqueeze(0).to(self.device)  # Add batch dimension
        elif self.feature_type == 'ce_kl':
            features = self.preprocessor.extract_features_by_sample(audio)
            features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        return features

    def recommend_two_stage(self):
        # ... (check if already done is fine) ...
        file_rec = {}
        for file_path in tqdm(list(Path(self.noisy_speech_dir).glob("*.wav")), desc="Recommending"):
            
            # --- FEATURE EXTRACTION ---
            waveform, sr, spec_features, cekl_features = self.get_features(file_path)

            # --- TWO-STAGE PREDICTION LOGIC ---
            with torch.no_grad():
                # 1. Gatekeeper prediction (always uses spec_features)
                gatekeeper_logits = self.gatekeeper_model(spec_features) if not self.use_hybrid_features else self.gatekeeper_model((spec_features, cekl_features))
                _, gatekeeper_prediction = torch.max(gatekeeper_logits, 1)

                if gatekeeper_prediction.item() == 1: # IS_SGMSE
                    final_prediction_idx = 2
                else: # NOT_SGMSE
                    # 2. Expert prediction (uses correct features based on flag)
                    if self.use_hybrid_features:
                        expert_input = (spec_features, cekl_features)
                    else:
                        expert_input = spec_features
                    
                    expert_logits = self.expert_model(expert_input)
                    _, expert_prediction = torch.max(expert_logits, 1)
                    
                    final_prediction_idx = self.expert_prediction_map_inv[expert_prediction.item()]
            
            file_rec[file_path] = final_prediction_idx
        breakpoint()
        rec_summary = {key: 0 for key in self.label_map.values()}
        for rec in file_rec.values():
            rec_summary[self.label_map.get(rec, 'Unknown')] += 1
        print("Recommendation Summary:")
        for rec, count in rec_summary.items():
            print(f"{rec}: {count} files")
        
        # based on recommendation, create subdirs
        breakpoint()
        for file, rec in file_rec.items():
            model_name = self.label_map.get(rec, 'Unknown')
            rec_dir = self.noisy_speech_dir + f"/rec_{model_name}"
            os.mkdir(rec_dir) if not os.path.exists(rec_dir) else None
            # rec_dir.mkdir(parents=True, exist_ok=True)
            # copy file to the recommended directory
            target_path = rec_dir + f'/{file.name}'
            if waveform.ndim > 1:
                waveform = torch.mean(waveform, dim=0)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)  
            torchaudio.save(target_path, waveform, sr)

    def recommend(self):
        breakpoint()
        # if Path(self.output_dir).glob("rec_*"):
        #     print("Recommendation already done, skipping...")
        #     return

        file_rec = {}
        for file in Path(self.noisy_speech_dir).glob("*.wav"):
            audio, sr = torchaudio.load(file)
            # audio = audio.to(self.device)
            features = self.extract_features(audio, sr)
            recommendation = self.model(features)
            recommendation_prob = torch.softmax(recommendation, dim=1)
            recommendation_idx = torch.argmax(recommendation_prob, dim=1)
            file_rec[file] = recommendation_idx.item()
            # print(f"File: {file}, Recommendation: {self.label_map.get(recommendation_idx.item(), 'Unknown')}, Probability: {recommendation_prob}")
        # summarize recommendation results
        breakpoint()
        rec_summary = {key: 0 for key in self.label_map.values()}
        for rec in file_rec.values():
            rec_summary[self.label_map.get(rec, 'Unknown')] += 1
        print("Recommendation Summary:")
        for rec, count in rec_summary.items():
            print(f"{rec}: {count} files")
        
        # based on recommendation, create subdirs
        breakpoint()
        for file, rec in file_rec.items():
            model_name = self.label_map.get(rec, 'Unknown')
            rec_dir = self.noisy_speech_dir + f"/rec_{model_name}"
            os.mkdir(rec_dir) if not os.path.exists(rec_dir) else None
            # rec_dir.mkdir(parents=True, exist_ok=True)
            # copy file to the recommended directory
            target_path = rec_dir + f'/{file.name}'
            torchaudio.save(target_path, audio.cpu(), sr)

    def enhance(self):
        breakpoint()
        for rec_dir in Path(self.output_dir).glob("rec_*"):
            if rec_dir.name == "rec_All_Failed":
                enhancement_pipeline = SpeechEnhancementPipeline(
                    noisy_speech_dir=rec_dir,
                    enhanced_dir=rec_dir / "enhanced",
                    device=self.device,
                )
                model_name = rec_dir.name.split("_")[-1]
                if model_name == 'Unknown' or  model_name == 'Failed':
                    model_name = 'SGMSE' # default model

                enhancement_pipeline.run_enhancement(
                    model_name=model_name
                )
        

class Evaluation:
    def __init__(self, enhanced_speech_dir, noisy_speech_dir, clean_speech_dir=None):
        self.enhanced_speech_dir = Path(enhanced_speech_dir)
        self.noisy_speech_dir = Path(noisy_speech_dir)
        self.clean_speech_dir = Path(clean_speech_dir) if clean_speech_dir else None
        self.pesq = PerceptualEvaluationSpeechQuality(fs=16000,mode='wb')
        self.stoi = ShortTimeObjectiveIntelligibility(fs=16000)
        self.dnsmos = DeepNoiseSuppressionMeanOpinionScore(fs=16000, personalized=False)

    def evaluate(self):
        results = []
        
        # Iterate through the noisy files as the "source of truth" for filenames
        for noisy_file in tqdm(list(self.noisy_speech_dir.glob("*.wav")), desc=f"Evaluating {self.enhanced_speech_dir.name}"):
            file_name = noisy_file.name
            
            try:
                # 1. Load Noisy Audio
                noisy_audio, sr_noi = torchaudio.load(noisy_file)
                if sr_noi != 16000:
                    noisy_audio = torchaudio.functional.resample(noisy_audio, orig_freq=sr_noi, new_freq=16000)

                # 2. Find and Load Corresponding Enhanced Audio
                enhanced_path = self.enhanced_speech_dir / file_name
                if not enhanced_path.exists():
                    print(f"Warning: Enhanced file not found for {file_name}, skipping.")
                    continue
                enhanced_audio, sr_enh = torchaudio.load(enhanced_path) # CORRECTED: load from enhanced_path
                if sr_enh != 16000:
                    enhanced_audio = torchaudio.functional.resample(enhanced_audio, orig_freq=sr_enh, new_freq=16000)

                # 3. Find and Load Corresponding Clean Audio (if provided)
                clean_audio = None
                if self.clean_speech_dir and self.clean_speech_dir.exists():
                    clean_path = self.clean_speech_dir / file_name
                    if clean_path.exists():
                        clean_audio, sr_cln = torchaudio.load(clean_path)
                        if sr_cln != 16000:
                            clean_audio = torchaudio.functional.resample(clean_audio, orig_freq=sr_cln, new_freq=16000)
                
                # 4. Calculate Metrics
                pesq_score = torch.tensor(0.0)
                stoi_score = torch.tensor(0.0)
                if clean_audio is not None:
                    # Ensure same length for PESQ/STOI - pad the shorter one
                    if enhanced_audio.shape[1] != clean_audio.shape[1]:
                        max_len = max(enhanced_audio.shape[1], clean_audio.shape[1])
                        enhanced_audio = torchaudio.functional.pad(enhanced_audio, (0, max_len - enhanced_audio.shape[1]))
                        clean_audio = torchaudio.functional.pad(clean_audio, (0, max_len - clean_audio.shape[1]))
                    
                    pesq_score = self.pesq(enhanced_audio, clean_audio)
                    stoi_score = self.stoi(enhanced_audio, clean_audio)
                
                dnsmos_scores = self.dnsmos(enhanced_audio.squeeze(0)) # DNSMOS expects 1D input

                results.append({
                    'file': file_name,
                    'pesq': pesq_score.item(),
                    'stoi': stoi_score.item(),
                    'dnsmos_0': dnsmos_scores[0].item(),  # DNSMOS 0
                    'dnsmos_sig': dnsmos_scores[1].item(),
                    'dnsmos_bak': dnsmos_scores[2].item(),
                    'dnsmos_ovr': dnsmos_scores[3].item()
                })
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
                continue

        # calculate mean and standard deviation of scores
        mean_pesq = np.mean([result['pesq'] for result in results])
        mean_stoi = np.mean([result['stoi'] for result in results])
        mean_dnsmos_0 = np.mean([result['dnsmos_0'] for result in results])
        mean_dnsmos_sig = np.mean([result['dnsmos_sig'] for result in results])
        mean_dnsmos_bak = np.mean([result['dnsmos_bak'] for result in results])
        mean_dnsmos_ovr = np.mean([result['dnsmos_ovr'] for result in results])

        std_pesq = np.std([result['pesq'] for result in results])
        std_stoi = np.std([result['stoi'] for result in results])
        std_dnsmos_0 = np.std([result['dnsmos_0'] for result in results])
        std_dnsmos_sig = np.std([result['dnsmos_sig'] for result in results])
        std_dnsmos_bak = np.std([result['dnsmos_bak'] for result in results])
        std_dnsmos_ovr = np.std([result['dnsmos_ovr'] for result in results])

        print(f"Mean PESQ: {mean_pesq}, Std PESQ: {std_pesq}")
        print(f"Mean STOI: {mean_stoi}, Std STOI: {std_stoi}")
        print(f"Mean DNSMOS 0: {mean_dnsmos_0}, Std DNSMOS: {std_dnsmos_0}")
        print(f"Mean DNSMOS Sig: {mean_dnsmos_sig}, Std DNSMOS Sig: {std_dnsmos_sig}")
        print(f"Mean DNSMOS Bak: {mean_dnsmos_bak}, Std DNSMOS Bak: {std_dnsmos_bak}")
        print(f"Mean DNSMOS Ovr: {mean_dnsmos_ovr}, Std DNSMOS Ovr: {std_dnsmos_ovr}")
        return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Recommender based speech enhancement")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the recommender model")
    parser.add_argument("--noisy_speech_dir", type=str, required=True, help="Directory containing noisy speech files")
    parser.add_argument("--enhanced_dir", type=str, required=True, help="Directory to save enhanced speech files")
    parser.add_argument("--clean_speech_dir", type=str, default=None, help="Directory to save output files")
    parser.add_argument("--feature_type", type=str, choices=['mel_spec', 'ce_kl'], default='mel_spec', help="Type of features to extract")
    parser.add_argument("--pretrained_models", type=str, default="resnet18", help="Pretrained model backbone for the recommender",
                        choices=["resnet18", "efficientnet_b0","ast_pretrain_audioset","ast_two_stage"])
    parser.add_argument("--gatekeeper_path", type=str, default=None, help="Path to the gatekeeper model")
    parser.add_argument("--expert_path", type=str, default=None, help="Path to the expert model")
    parser.add_argument('--eval_only', action='store_true', help="Run evaluation only without enhancement")
    parser.add_argument('--use_hybrid_features', action='store_true', help="Use hybrid features for recommendation")
    parser.add_argument('--cekl_features_base_dir', type=str, default=None, help="Base directory for CE/KL features if using hybrid expert")
    
    args = parser.parse_args()
    breakpoint()
    if args.eval_only:
        all_results = []
        for noisy_speech_dir in Path(args.noisy_speech_dir).glob("rec_*"):
            breakpoint()
            if not noisy_speech_dir.is_dir():
                continue
            model_name = noisy_speech_dir.name.split("_")[-1]
            if model_name == 'Unknown' or model_name == 'Failed':
                model_name = 'SGMSE'                
            enhanced_speech_dir = Path(args.enhanced_dir) / model_name
            # if model_name == 'CDiffuSE':
            #     enhanced_speech_dir = '/success_fail_estimation/noisy_blind_testset_v3_challenge_withSNR_16k/enhanced/Enhanced/CDiffuSE/model370200/test/spec'
            eval = Evaluation(
                enhanced_speech_dir=enhanced_speech_dir,
                noisy_speech_dir=noisy_speech_dir,
                clean_speech_dir=Path(args.clean_speech_dir) if args.clean_speech_dir else None
            )
            results = eval.evaluate()
            all_results.extend(results)
        breakpoint()
        # calculate mean and std of all results
        mean_pesq = np.mean([result['pesq'] for result in all_results])
        mean_stoi = np.mean([result['stoi'] for result in all_results])
        mean_dnsmos_0 = np.mean([result['dnsmos_0'] for result in all_results])
        mean_dnsmos_sig = np.mean([result['dnsmos_sig'] for result in all_results])
        mean_dnsmos_bak = np.mean([result['dnsmos_bak'] for result in all_results])
        mean_dnsmos_ovr = np.mean([result['dnsmos_ovr'] for result in all_results])
        std_pesq = np.std([result['pesq'] for result in all_results])
        std_stoi = np.std([result['stoi'] for result in all_results])
        std_dnsmos_0 = np.std([result['dnsmos_0'] for result in all_results])
        std_dnsmos_sig = np.std([result['dnsmos_sig'] for result in all_results])
        std_dnsmos_bak = np.std([result['dnsmos_bak'] for result in all_results])
        std_dnsmos_ovr = np.std([result['dnsmos_ovr'] for result in all_results])
        print(f"Overall Mean PESQ: {mean_pesq}, Std PESQ: {std_pesq}")
        print(f"Overall Mean STOI: {mean_stoi}, Std STOI: {std_stoi}")
        print(f"Overall Mean DNSMOS 0: {mean_dnsmos_0}, Std DNSMOS: {std_dnsmos_0}")
        print(f"Overall Mean DNSMOS Sig: {mean_dnsmos_sig}, Std DNSMOS Sig: {std_dnsmos_sig}")
        print(f"Overall Mean DNSMOS Bak: {mean_dnsmos_bak}, Std DNSMOS Bak: {std_dnsmos_bak}")
        print(f"Overall Mean DNSMOS Ovr: {mean_dnsmos_ovr}, Std DNSMOS Ovr: {std_dnsmos_ovr}")

        # for result in results:
        #     print(f"File: {result['file']}, PESQ: {result['pesq']}, STOI: {result['stoi']}, DNSMOS: {result['dnsmos']}")
    else:
        recommender = RecommenderBasedEnhancement(
            model_path=args.model_path if args.model_path is not None else None,
            noisy_speech_dir=args.noisy_speech_dir,
            output_dir=args.enhanced_dir,
            gatekeeper_path=args.gatekeeper_path,
            expert_path=args.expert_path,
            use_hybrid_features=args.use_hybrid_features,
            cekl_features_base_dir=args.cekl_features_base_dir,
            feature_type=args.feature_type,
        )
        if args.gatekeeper_path and args.expert_path:
            recommender.recommend_two_stage()
        else:
            recommender.recommend()
        recommender.enhance()