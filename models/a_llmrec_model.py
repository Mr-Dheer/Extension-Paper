# a_llmrec_model_clip.py
# A-LLMRec with CLIP embeddings + SmolVLM2-2.2B-Instruct
# 
# Architecture:
# Stage 1: Align CF embeddings (SASRec) with CLIP embeddings in shared latent space
# Stage 2: Train projection layers to map embeddings to SmolVLM2 token space
#
# Key dimensions:
# - SASRec item embeddings: 50-dim
# - CLIP fused embeddings: 1536-dim (768 text + 768 image)
# - Shared latent space: 128-dim
# - SmolVLM2 hidden size: 2048-dim

import os
import random
import pickle
import torch
from torch.cuda.amp import autocast
import torch.nn as nn
import numpy as np

from models.recsys_model import RecSys


def create_dir(path):
    """Create directory if not exists"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[DIR] Created: {path}")


class TwoLayerMLP(nn.Module):
    """
    Two-layer MLP for embedding alignment.
    Architecture: input_dim -> latent_dim -> input_dim
    Outputs both latent representation and reconstruction.
    """
    def __init__(self, input_dim, latent_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, input_dim)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        latent = self.activation(self.fc1(x))
        reconstructed = self.fc2(latent)
        return latent, reconstructed


class A_llmrec_model(nn.Module):
    """
    A-LLMRec model with CLIP multimodal embeddings and SmolVLM2 LLM.
    
    Two-stage training:
    1. Stage 1: Align CF (collaborative filtering) embeddings with CLIP embeddings
       - CF embeddings from pretrained SASRec
       - CLIP embeddings precomputed (text + image fused)
       - Learn shared 128-dim latent space
       
    2. Stage 2: Train projection to SmolVLM2 token space
       - Project user log embedding to LLM hidden space (2048)
       - Project item embeddings to LLM hidden space
       - Train with next-item prediction task
    """
    
    def __init__(self, args):
        super().__init__()
        
        print("\n" + "="*70)
        print("INITIALIZING A-LLMRec WITH CLIP + SmolVLM2")
        print("="*70)
        
        self.args = args
        self.device = args.device
        
        # Log configuration
        print(f"[CONFIG] Dataset: {args.rec_pre_trained_data}")
        print(f"[CONFIG] Device: {self.device}")
        print(f"[CONFIG] RecSys: {args.recsys}")
        print(f"[CONFIG] Stage 1: {getattr(args, 'pretrain_stage1', False)}")
        print(f"[CONFIG] Stage 2: {getattr(args, 'pretrain_stage2', False)}")
        print(f"[CONFIG] Inference: {getattr(args, 'inference', False)}")
        
        # === LOAD TEXT DICTIONARY (for prompts) ===
        text_dict_path = f'./data/amazon/{args.rec_pre_trained_data}_text_name_dict.json.gz'
        print(f"[LOAD] Text dictionary: {text_dict_path}")
        with open(text_dict_path, 'rb') as f:
            self.text_name_dict = pickle.load(f)
        print(f"[LOAD] Loaded {len(self.text_name_dict.get('title', {}))} item titles")
        
        # === LOAD RECSYS (SASRec) ===
        print(f"[LOAD] Loading RecSys: {args.recsys}")
        self.recsys = RecSys(args.recsys, args.rec_pre_trained_data, self.device)
        self.item_num = self.recsys.item_num
        self.rec_sys_dim = self.recsys.hidden_units  # Usually 50
        print(f"[RECSYS] Items: {self.item_num}, Embedding dim: {self.rec_sys_dim}")
        
        # === DIMENSION CONFIGURATION ===
        # self.clip_dim = 1536      # CLIP fused (768 text + 768 image)
        self.clip_dim = 2048      # CLIP fused (768 text + 768 image)
        self.latent_dim = 128     # Shared latent space
        self.llm_hidden_size = 2048  # SmolVLM2-2.2B hidden size
        
        print(f"[DIM] CLIP: {self.clip_dim}")
        print(f"[DIM] Latent: {self.latent_dim}")
        print(f"[DIM] RecSys: {self.rec_sys_dim}")
        print(f"[DIM] LLM hidden: {self.llm_hidden_size}")
        
        # === MLP FOR CF EMBEDDINGS ===
        # Maps: rec_sys_dim -> latent_dim -> rec_sys_dim
        self.mlp = TwoLayerMLP(self.rec_sys_dim, self.latent_dim)
        print(f"[MLP] CF encoder: {self.rec_sys_dim} -> {self.latent_dim} -> {self.rec_sys_dim}")
        
        # === CLIP EMBEDDINGS (Stage 1) ===
        if getattr(args, 'pretrain_stage1', False):
            self._load_clip_embeddings(args)
            
            # MLP for CLIP embeddings: clip_dim -> latent_dim -> clip_dim
            self.mlp2 = TwoLayerMLP(self.clip_dim, self.latent_dim)
            print(f"[MLP] CLIP encoder: {self.clip_dim} -> {self.latent_dim} -> {self.clip_dim}")
        
        # === LOSS FUNCTIONS ===
        self.mse = nn.MSELoss()
        self.bce_criterion = nn.BCEWithLogitsLoss()
        
        # === TRACKING ===
        self.maxlen = args.maxlen
        self.NDCG = 0
        self.HIT = 0
        self.num_user = 0
        
        # === LLM FOR STAGE 2 ===
        if getattr(args, 'pretrain_stage2', False) or getattr(args, 'inference', False):
            self._init_llm_components()
        
        print("="*70)
        print("INITIALIZATION COMPLETE")
        print("="*70 + "\n")

    def _load_clip_embeddings(self, args):
        """Load precomputed CLIP embeddings"""
        clip_path = getattr(args, 'clip_emb_path', 
            '/home/kavach/Dev/Extension-Paper/Clip/ALIGNED_ALLM_PATCHED/clip_fused_aligned_lion_huge.npy')
        
        print(f"[CLIP] Loading from: {clip_path}")
        
        if not os.path.exists(clip_path):
            raise FileNotFoundError(f"CLIP embeddings not found: {clip_path}")
        
        clip_emb = np.load(clip_path)
        self.register_buffer('clip_embeddings', torch.tensor(clip_emb, dtype=torch.float32))
        
        print(f"[CLIP] Shape: {self.clip_embeddings.shape}")
        print(f"[CLIP] Dtype: {self.clip_embeddings.dtype}")
        
        # Verify dimension
        actual_dim = self.clip_embeddings.shape[1]
        if actual_dim != self.clip_dim:
            print(f"[CLIP] WARNING: Expected {self.clip_dim}, got {actual_dim}. Updating clip_dim.")
            self.clip_dim = actual_dim
        
        # Verify alignment
        expected = self.item_num + 1  # +1 for padding at index 0
        actual = self.clip_embeddings.shape[0]
        if actual != expected:
            print(f"[CLIP] WARNING: Expected {expected} items, got {actual}")
            if actual < expected:
                print(f"[CLIP] ERROR: Missing embeddings for {expected - actual} items!")
        else:
            print(f"[CLIP] ✓ Aligned: {actual} embeddings")
        
        # Check padding row
        padding_norm = torch.norm(self.clip_embeddings[0]).item()
        print(f"[CLIP] Padding (row 0) norm: {padding_norm:.6f} (should be ~0)")
        
        # Statistics
        norms = torch.norm(self.clip_embeddings[1:], dim=1)
        print(f"[CLIP] Item norms - Mean: {norms.mean():.4f}, Std: {norms.std():.4f}")

    def _init_llm_components(self):
        """Initialize LLM and projection layers for Stage 2"""
        from models.llm4rec import llm4rec
        
        print("\n[LLM] Initializing SmolVLM2...")
        self.llm = llm4rec(device=self.device)
        
        # Get actual hidden size from LLM
        self.llm_hidden_size = self.llm.hidden_size
        print(f"[LLM] Hidden size: {self.llm_hidden_size}")
        
        # User log embedding projection: rec_sys_dim -> llm_hidden
        self.log_emb_proj = nn.Sequential(
            nn.Linear(self.rec_sys_dim, self.llm_hidden_size),
            nn.LayerNorm(self.llm_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.llm_hidden_size, self.llm_hidden_size)
        )
        nn.init.xavier_normal_(self.log_emb_proj[0].weight)
        nn.init.xavier_normal_(self.log_emb_proj[3].weight)
        print(f"[PROJ] log_emb_proj: {self.rec_sys_dim} -> {self.llm_hidden_size}")
        
        # Item embedding projection: latent_dim -> llm_hidden
        self.item_emb_proj = nn.Sequential(
            nn.Linear(self.latent_dim, self.llm_hidden_size),
            nn.LayerNorm(self.llm_hidden_size),
            nn.GELU(),
            nn.Linear(self.llm_hidden_size, self.llm_hidden_size)
        )
        nn.init.xavier_normal_(self.item_emb_proj[0].weight)
        nn.init.xavier_normal_(self.item_emb_proj[3].weight)
        print(f"[PROJ] item_emb_proj: {self.latent_dim} -> {self.llm_hidden_size}")

    def get_clip_embedding(self, item_ids):
        """
        Get CLIP embeddings for item IDs.
        
        Args:
            item_ids: numpy array, list, or tensor of item IDs (1-indexed, 0=padding)
        Returns:
            torch.Tensor [batch_size, clip_dim]
        """
        if isinstance(item_ids, np.ndarray):
            indices = torch.from_numpy(item_ids).long()
        elif isinstance(item_ids, list):
            indices = torch.LongTensor(item_ids)
        elif isinstance(item_ids, torch.Tensor):
            indices = item_ids.long()
        else:
            indices = torch.LongTensor([item_ids])
        
        indices = indices.to(self.device)
        
        # Clamp to valid range
        max_idx = self.clip_embeddings.shape[0] - 1
        indices = torch.clamp(indices, 0, max_idx)
        
        return self.clip_embeddings[indices]

    def get_item_emb(self, item_ids):
        """
        Get joint collaborative-text embedding (Stage 2).
        Returns 128-dim latent from trained CF MLP.
        
        Args:
            item_ids: list or array of item IDs
        Returns:
            torch.Tensor [num_items, latent_dim]
        """
        with torch.no_grad():
            item_tensor = torch.LongTensor(item_ids).to(self.device)
            item_embs = self.recsys.model.item_emb(item_tensor)
            latent, _ = self.mlp(item_embs)
        return latent

    def save_model(self, args, epoch1=None, epoch2=None):
        """Save model checkpoints"""
        out_dir = './models/saved_models/'
        create_dir(out_dir)
        
        base = f'{args.rec_pre_trained_data}_{args.recsys}_{epoch1}_clip_'
        
        if getattr(args, 'pretrain_stage1', False):
            torch.save(self.mlp.state_dict(), out_dir + base + 'mlp.pt')
            torch.save(self.mlp2.state_dict(), out_dir + base + 'mlp2_clip.pt')
            print(f"[SAVE] Stage 1: {out_dir + base}mlp.pt, mlp2_clip.pt")
        
        if getattr(args, 'pretrain_stage2', False):
            s2_base = out_dir + base + f'smolvlm2_{epoch2}_'
            torch.save(self.log_emb_proj.state_dict(), s2_base + 'log_proj.pt')
            torch.save(self.item_emb_proj.state_dict(), s2_base + 'item_proj.pt')
            print(f"[SAVE] Stage 2: {s2_base}log_proj.pt, item_proj.pt")

    def load_model(self, args, phase1_epoch=None, phase2_epoch=None):
        """Load model checkpoints"""
        print("\n[LOAD] Loading model checkpoints...")
        
        out_dir = './models/saved_models/'
        base = f'{args.rec_pre_trained_data}_{args.recsys}_{phase1_epoch}_clip_'
        
        # Stage 1: CF MLP
        mlp_path = out_dir + base + 'mlp.pt'
        if os.path.exists(mlp_path):
            self.mlp.load_state_dict(torch.load(mlp_path, map_location=args.device))
            for p in self.mlp.parameters():
                p.requires_grad = False
            print(f"[LOAD] ✓ {mlp_path}")
        else:
            print(f"[LOAD] ✗ Not found: {mlp_path}")
        
        # Stage 1: CLIP MLP (optional)
        mlp2_path = out_dir + base + 'mlp2_clip.pt'
        if os.path.exists(mlp2_path) and hasattr(self, 'mlp2'):
            self.mlp2.load_state_dict(torch.load(mlp2_path, map_location=args.device))
            for p in self.mlp2.parameters():
                p.requires_grad = False
            print(f"[LOAD] ✓ {mlp2_path}")
        
        # Stage 2 projections
        if getattr(args, 'inference', False) or getattr(args, 'pretrain_stage2', False):
            s2_base = out_dir + base + f'smolvlm2_{phase2_epoch}_'
            
            log_path = s2_base + 'log_proj.pt'
            if os.path.exists(log_path):
                self.log_emb_proj.load_state_dict(torch.load(log_path, map_location=args.device))
                print(f"[LOAD] ✓ {log_path}")
            
            item_path = s2_base + 'item_proj.pt'
            if os.path.exists(item_path):
                self.item_emb_proj.load_state_dict(torch.load(item_path, map_location=args.device))
                print(f"[LOAD] ✓ {item_path}")
        
        print("[LOAD] Complete\n")

    def find_item_text(self, items, title_flag=True, description_flag=True):
        """Get text for multiple items"""
        texts = []
        for item in items:
            texts.append(self.find_item_text_single(item, title_flag, description_flag))
        return texts

    def find_item_text_single(self, item, title_flag=True, description_flag=True):
        """Get text for single item"""
        title = self.text_name_dict.get('title', {}).get(item, 'No Title')
        desc = self.text_name_dict.get('description', {}).get(item, 'No Description')
        
        if title_flag and description_flag:
            return f'"{title}, {desc}"'
        elif title_flag:
            return f'"{title}"'
        else:
            return f'"{desc}"'

    def forward(self, data, optimizer=None, batch_iter=None, mode='phase1'):
        """Forward dispatch"""
        if mode == 'phase1':
            return self.pre_train_phase1(data, optimizer, batch_iter)
        elif mode == 'phase2':
            return self.pre_train_phase2(data, optimizer, batch_iter)
        elif mode == 'generate':
            return self.generate(data)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def pre_train_phase1(self, data, optimizer, batch_iter):
        """
        Stage 1: Align CF embeddings with CLIP embeddings.
        
        Loss components:
        1. L_rec (BPR): Recommendation loss
        2. L_match: Align CF latent with CLIP latent
        3. L_cf_recon: Reconstruct CF embeddings
        4. L_clip_recon: Reconstruct CLIP embeddings
        """
        epoch, total_epoch, step, total_step = batch_iter
        
        u, seq, pos, neg = data
        batch_size = u.shape[0]
        
        # Get indices for last position in sequence
        indices = [self.maxlen * (i + 1) - 1 for i in range(batch_size)]
        
        # Get CF embeddings (frozen)
        with torch.no_grad():
            log_emb_full, pos_emb_full, neg_emb_full = self.recsys.model(
                u, seq, pos, neg, mode='item'
            )
        
        # Extract last position
        log_emb_ = log_emb_full[indices]
        pos_emb_ = pos_emb_full[indices]
        neg_emb_ = neg_emb_full[indices]
        pos_ids = pos.reshape(-1)[indices]
        neg_ids = neg.reshape(-1)[indices]
        
        # Mini-batch processing
        mini_batch = 60
        start = 0
        n_iter = 0
        
        loss_total, loss_bpr, loss_match, loss_cf, loss_clip = 0, 0, 0, 0, 0
        
        while start < len(log_emb_):
            end = min(start + mini_batch, len(log_emb_))
            optimizer.zero_grad()
            
            # Slice
            log_e = log_emb_[start:end]
            pos_e = pos_emb_[start:end]
            neg_e = neg_emb_[start:end]
            pos_i = pos_ids[start:end]
            neg_i = neg_ids[start:end]
            
            # Get CLIP embeddings
            pos_clip = self.get_clip_embedding(pos_i)
            neg_clip = self.get_clip_embedding(neg_i)
            
            # CF MLP forward
            pos_cf_lat, pos_cf_rec = self.mlp(pos_e)
            neg_cf_lat, neg_cf_rec = self.mlp(neg_e)
            
            # CLIP MLP forward
            pos_clip_lat, pos_clip_rec = self.mlp2(pos_clip)
            neg_clip_lat, neg_clip_rec = self.mlp2(neg_clip)
            
            # Loss 1: Recommendation (BPR-style)
            pos_logits = (log_e * pos_cf_rec).mean(dim=1)
            neg_logits = (log_e * neg_cf_rec).mean(dim=1)
            l_bpr = self.bce_criterion(pos_logits, torch.ones_like(pos_logits))
            l_bpr += self.bce_criterion(neg_logits, torch.zeros_like(neg_logits))
            
            # Loss 2: Matching (align latent spaces)
            l_match = self.mse(pos_cf_lat, pos_clip_lat)
            l_match += self.mse(neg_cf_lat, neg_clip_lat)
            
            # Loss 3: CF reconstruction
            l_cf = self.mse(pos_cf_rec, pos_e) + self.mse(neg_cf_rec, neg_e)
            
            # Loss 4: CLIP reconstruction
            l_clip = self.mse(pos_clip_rec, pos_clip.detach())
            l_clip += self.mse(neg_clip_rec, neg_clip.detach())
            
            # Total (paper coefficients: α=0.5, β=0.2)
            total = l_bpr + l_match + 0.5 * l_cf + 0.2 * l_clip
            
            total.backward()
            optimizer.step()
            
            loss_total += total.item()
            loss_bpr += l_bpr.item()
            loss_match += l_match.item()
            loss_cf += l_cf.item()
            loss_clip += l_clip.item()
            
            n_iter += 1
            start = end
        
        # Log
        if n_iter > 0:
            print(f"[S1] E{epoch}/{total_epoch} S{step}/{total_step} | "
                  f"Tot: {loss_total/n_iter:.4f} | "
                  f"BPR: {loss_bpr/n_iter:.4f} | "
                  f"Match: {loss_match/n_iter:.4f} | "
                  f"CF: {loss_cf/n_iter:.4f} | "
                  f"CLIP: {loss_clip/n_iter:.4f}")
        
        return loss_total / max(n_iter, 1)

    def make_interact_text(self, interact_ids, max_num):
        """Create interaction history text with [HistoryEmb] tokens"""
        titles = self.find_item_text(interact_ids, title_flag=True, description_flag=False)
        
        if max_num != 'all':
            titles = titles[-max_num:]
            interact_ids = interact_ids[-max_num:]
        
        text = ','.join([t + '[HistoryEmb]' for t in titles])
        return text, interact_ids

    def make_candidate_text(self, interact_ids, num_candidates, target_id, target_title):
        """Create candidate text with [CandidateEmb] tokens"""
        # Sample negatives
        interact_set = set(interact_ids.tolist() if isinstance(interact_ids, np.ndarray) else interact_ids)
        neg_ids = []
        while len(neg_ids) < 50:
            t = np.random.randint(1, self.item_num + 1)
            if t not in interact_set and t not in neg_ids and t != target_id:
                neg_ids.append(t)
        
        # Build candidates
        cand_ids = [target_id] + neg_ids[:num_candidates - 1]
        cand_texts = [target_title + '[CandidateEmb]']
        
        for cid in neg_ids[:num_candidates - 1]:
            cand_texts.append(
                self.find_item_text_single(cid, title_flag=True, description_flag=False) + '[CandidateEmb]'
            )
        
        # Shuffle
        perm = np.random.permutation(len(cand_ids))
        cand_ids = np.array(cand_ids)[perm]
        cand_texts = np.array(cand_texts)[perm]
        
        return ','.join(cand_texts), cand_ids

    def pre_train_phase2(self, data, optimizer, batch_iter):
        """
        Stage 2: Train LLM projection layers.
        """
        epoch, total_epoch, step, total_step = batch_iter
        optimizer.zero_grad()
        
        u, seq, pos, neg = data
        batch_size = len(u)
        
        text_input, text_output = [], []
        interact_embs, candidate_embs = [], []
        
        self.llm.eval()
        
        # Get user representations
        with torch.no_grad():
            log_emb = self.recsys.model(u, seq, pos, neg, mode='log_only')
        
        for i in range(batch_size):
            target_id = pos[i][-1]
            target_title = self.find_item_text_single(target_id, title_flag=True, description_flag=False)
            
            # History
            valid_seq = seq[i][seq[i] > 0]
            interact_text, interact_ids = self.make_interact_text(valid_seq, 10)
            
            # Candidates
            cand_text, cand_ids = self.make_candidate_text(valid_seq, 20, target_id, target_title)
            
            # Build prompt (SmolVLM2 style)
            prompt = '[UserRep] is a user representation. '
            
            # Dataset-specific verb
            dataset = self.args.rec_pre_trained_data
            if dataset == 'Movies_and_TV':
                prompt += 'This user has watched '
            elif dataset == 'Video_Games':
                prompt += 'This user has played '
            else:
                prompt += 'This user has bought '
            
            prompt += interact_text + ' in the past. '
            
            # Instruction
            if dataset == 'Movies_and_TV':
                prompt += 'Recommend one movie from: '
            elif dataset == 'Video_Games':
                prompt += 'Recommend one game from: '
            else:
                prompt += 'Recommend one item from: '
            
            prompt += cand_text + '. The recommendation is '
            
            text_input.append(prompt)
            text_output.append(target_title)
            
            # Get embeddings
            interact_embs.append(self.item_emb_proj(self.get_item_emb(interact_ids)))
            candidate_embs.append(self.item_emb_proj(self.get_item_emb(cand_ids)))
        
        # Prepare samples
        samples = {
            'text_input': text_input,
            'text_output': text_output,
            'interact': interact_embs,
            'candidate': candidate_embs
        }
        
        # Project user representation
        log_emb_proj = self.log_emb_proj(log_emb).to(dtype=torch.bfloat16)

        
        # Forward through LLM
        loss = self.llm(log_emb_proj, samples)
        
        loss.backward()
        optimizer.step()
        
        print(f"[S2] E{epoch}/{total_epoch} S{step}/{total_step} | Loss: {loss.item():.4f}")
        return loss.item()

    def generate(self, data):
        """Generate recommendations"""
        u, seq, pos, neg, rank = data
        batch_size = len(u)
        
        answers, text_inputs = [], []
        interact_embs, candidate_embs = [], []
        
        with torch.no_grad():
            log_emb = self.recsys.model(u, seq, pos, neg, mode='log_only')
            
            for i in range(batch_size):
                target_id = pos[i]
                target_title = self.find_item_text_single(target_id, title_flag=True, description_flag=False)
                
                valid_seq = seq[i][seq[i] > 0]
                interact_text, interact_ids = self.make_interact_text(valid_seq, 10)
                cand_text, cand_ids = self.make_candidate_text(valid_seq, 20, target_id, target_title)
                
                # Build prompt
                prompt = '[UserRep] is a user representation. '
                dataset = self.args.rec_pre_trained_data
                
                if dataset == 'Movies_and_TV':
                    prompt += 'This user has watched '
                elif dataset == 'Video_Games':
                    prompt += 'This user has played '
                else:
                    prompt += 'This user has bought '
                
                prompt += interact_text + ' in the past. '
                
                if dataset == 'Movies_and_TV':
                    prompt += 'Recommend one movie from: '
                elif dataset == 'Video_Games':
                    prompt += 'Recommend one game from: '
                else:
                    prompt += 'Recommend one item from: '
                
                prompt += cand_text + '. The recommendation is '
                
                answers.append(target_title)
                text_inputs.append(prompt)
                
                interact_embs.append(self.item_emb_proj(self.get_item_emb(interact_ids)).to(dtype=torch.bfloat16))
                candidate_embs.append(self.item_emb_proj(self.get_item_emb(cand_ids)).to(dtype=torch.bfloat16))
        
        log_emb_proj = self.log_emb_proj(log_emb)
        
        samples = {
            'text_input': text_inputs,
            'interact': interact_embs,
            'candidate': candidate_embs
        }
        
        outputs = self.llm.generate(log_emb_proj, samples, max_new_tokens=64)
        
        # Log results
        out_file = './recommendation_output_smolvlm2_clip_lion_G_seed_1.txt'
        with open(out_file, 'a') as f:
            for i in range(batch_size):
                f.write("="*60 + "\n")
                f.write(f"Input: {text_inputs[i][:200]}...\n\n")
                f.write(f"Answer: {answers[i]}\n")
                f.write(f"Generated: {outputs[i]}\n\n")
        
        print(f"[GEN] Saved {batch_size} results to {out_file}")
        return outputs