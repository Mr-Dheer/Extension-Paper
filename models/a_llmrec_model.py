# a_llmrec_model_clip.py

import random
import pickle
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import numpy as np

from models.recsys_model import *
from models.llm4rec import *
# Remove: from sentence_transformers import SentenceTransformer


class two_layer_mlp(nn.Module):
    """MLP for CF embeddings: input_dim -> 128 -> input_dim"""
    def __init__(self, dims):
        super().__init__()
        self.fc1 = nn.Linear(dims, 128)
        self.fc2 = nn.Linear(128, dims)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x1 = self.fc2(x)
        return x, x1  # (latent_128, reconstructed)


class two_layer_mlp_clip(nn.Module):
    """MLP for CLIP embeddings: 512 -> 128 -> 512"""
    def __init__(self, input_dim=512, latent_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, input_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        latent = self.fc1(x)
        latent = self.sigmoid(latent)
        reconstructed = self.fc2(latent)
        return latent, reconstructed  # (latent_128, reconstructed_512)


class A_llmrec_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        rec_pre_trained_data = args.rec_pre_trained_data
        self.args = args
        self.device = args.device
        
        # Load text name dict (still needed for Stage 2 prompts)
        with open(f'./data/amazon/{args.rec_pre_trained_data}_text_name_dict.json.gz','rb') as ft:
            self.text_name_dict = pickle.load(ft)
        
        # Load CF-RecSys (frozen)
        self.recsys = RecSys(args.recsys, rec_pre_trained_data, self.device)
        self.item_num = self.recsys.item_num
        self.rec_sys_dim = self.recsys.hidden_units  # 50
        
        # === CLIP CONFIG (changed from SBERT) ===
        # Put the correct emb size over here
        self.clip_dim = 1536  # Changed from sbert_dim = 768
        self.latent_dim = 128
        
        # MLP for CF embeddings: 50 -> 128 -> 50
        self.mlp = two_layer_mlp(self.rec_sys_dim)
        
        if args.pretrain_stage1:
            # === LOAD PRE-COMPUTED CLIP EMBEDDINGS ===
            clip_path = '/home/kavach/Dev/Extension-Paper/Clip/ALIGNED_ALLM_PATCHED/clip_fused_aligned.npy'
            
            clip_emb_np = np.load(clip_path)
            self.register_buffer('clip_embeddings', 
                                torch.tensor(clip_emb_np, dtype=torch.float32))
            print(f"CLIP embeddings shape: {self.clip_embeddings.shape}")
            
            # Verify alignment
            expected_items = self.item_num + 1  # +1 for padding at index 0
            if self.clip_embeddings.shape[0] != expected_items:
                print(f"WARNING: CLIP embeddings ({self.clip_embeddings.shape[0]}) != "
                      f"expected items ({expected_items})")
            
            # MLP for CLIP embeddings: 512 -> 128 -> 512
            self.mlp2 = two_layer_mlp_clip(self.clip_dim, self.latent_dim)
            
            # Remove SBERT - no longer needed
            # self.sbert = SentenceTransformer('nq-distilbert-base-v1')
        
        self.mse = nn.MSELoss()
        
        self.maxlen = args.maxlen
        self.NDCG = 0
        self.HIT = 0
        self.rec_NDCG = 0
        self.rec_HIT = 0
        self.lan_NDCG = 0
        self.lan_HIT = 0
        self.num_user = 0
        self.yes = 0
        
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()
        
        if args.pretrain_stage2 or args.inference:
            self.llm = llm4rec(device=self.device)
            llm_hidden_size = self.llm.llm_model.config.text_config.hidden_size
            
            self.log_emb_proj = nn.Sequential(
                nn.Linear(self.rec_sys_dim, llm_hidden_size),
                nn.LayerNorm(llm_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(llm_hidden_size, llm_hidden_size)
            )
            nn.init.xavier_normal_(self.log_emb_proj[0].weight)
            nn.init.xavier_normal_(self.log_emb_proj[3].weight)

            self.item_emb_proj = nn.Sequential(
                nn.Linear(self.latent_dim, llm_hidden_size),  # 128 -> llm_hidden
                nn.LayerNorm(llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size)
            )
            nn.init.xavier_normal_(self.item_emb_proj[0].weight)
            nn.init.xavier_normal_(self.item_emb_proj[3].weight)

    def get_clip_embedding(self, item_ids):
        """
        Retrieve pre-computed CLIP embeddings for item IDs.
        
        Args:
            item_ids: numpy array or list of item IDs (1-indexed, 0=padding)
        Returns:
            torch.Tensor [len(item_ids), 512]
        """
        indices = torch.LongTensor(item_ids).to(self.device)
        # Clamp to valid range (safety check)
        indices = torch.clamp(indices, 0, self.clip_embeddings.shape[0] - 1)
        return self.clip_embeddings[indices]
            
    def save_model(self, args, epoch1=None, epoch2=None):
        out_dir = f'./models/saved_models/'
        create_dir(out_dir)
        out_dir += f'{args.rec_pre_trained_data}_{args.recsys}_{epoch1}_clip_'
        
        if args.pretrain_stage1:
            # Don't save SBERT anymore, save CLIP MLP instead
            torch.save(self.mlp.state_dict(), out_dir + 'mlp.pt')
            torch.save(self.mlp2.state_dict(), out_dir + 'mlp2_clip.pt')
        
        out_dir += f'{args.llm}_{epoch2}_'
        if args.pretrain_stage2:
            torch.save(self.log_emb_proj.state_dict(), out_dir + 'log_proj.pt')
            torch.save(self.item_emb_proj.state_dict(), out_dir + 'item_proj.pt')
            
    def load_model(self, args, phase1_epoch=None, phase2_epoch=None):
        out_dir = f'./models/saved_models/{args.rec_pre_trained_data}_{args.recsys}_{phase1_epoch}_clip_'
        
        mlp = torch.load(out_dir + 'mlp.pt', map_location=args.device)
        self.mlp.load_state_dict(mlp)
        del mlp
        for name, param in self.mlp.named_parameters():
            param.requires_grad = False

        if args.inference:
            out_dir += f'{args.llm}_{phase2_epoch}_'
            
            log_emb_proj_dict = torch.load(out_dir + 'log_proj.pt', map_location=args.device)
            self.log_emb_proj.load_state_dict(log_emb_proj_dict)
            del log_emb_proj_dict
            
            item_emb_proj_dict = torch.load(out_dir + 'item_proj.pt', map_location=args.device)
            self.item_emb_proj.load_state_dict(item_emb_proj_dict)
            del item_emb_proj_dict

    def find_item_text(self, item, title_flag=True, description_flag=True):
        """Still needed for Stage 2 prompts"""
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'
        if title_flag and description_flag:
            return [f'"{self.text_name_dict[t].get(i,t_)}, {self.text_name_dict[d].get(i,d_)}"' for i in item]
        elif title_flag and not description_flag:
            return [f'"{self.text_name_dict[t].get(i,t_)}"' for i in item]
        elif not title_flag and description_flag:
            return [f'"{self.text_name_dict[d].get(i,d_)}"' for i in item]
    
    def find_item_text_single(self, item, title_flag=True, description_flag=True):
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'
        if title_flag and description_flag:
            return f'"{self.text_name_dict[t].get(item,t_)}, {self.text_name_dict[d].get(item,d_)}"'
        elif title_flag and not description_flag:
            return f'"{self.text_name_dict[t].get(item,t_)}"'
        elif not title_flag and description_flag:
            return f'"{self.text_name_dict[d].get(item,d_)}"'
        
    def get_item_emb(self, item_ids):
        """Get joint collaborative-CLIP embedding (used in Stage 2)"""
        with torch.no_grad():
            item_embs = self.recsys.model.item_emb(torch.LongTensor(item_ids).to(self.device))
            item_embs, _ = self.mlp(item_embs)  # Returns 128-dim latent
        return item_embs
    
    def forward(self, data, optimizer=None, batch_iter=None, mode='phase1'):
        if mode == 'phase1':
            self.pre_train_phase1(data, optimizer, batch_iter)
        if mode == 'phase2':
            self.pre_train_phase2(data, optimizer, batch_iter)
        if mode == 'generate':
            self.generate(data)

    def pre_train_phase1(self, data, optimizer, batch_iter):
        """
        Stage 1: Align CF embeddings with CLIP embeddings
        
        Original: SBERT(title+desc) computed on-the-fly
        Modified: CLIP embeddings loaded from pre-computed .npy
        """
        epoch, total_epoch, step, total_step = batch_iter
        
        # No more self.sbert.train()
        optimizer.zero_grad()

        u, seq, pos, neg = data
        indices = [self.maxlen * (i + 1) - 1 for i in range(u.shape[0])]
        
        # Get CF embeddings (frozen)
        with torch.no_grad():
            log_emb, pos_emb, neg_emb = self.recsys.model(u, seq, pos, neg, mode='item')
            
        log_emb_ = log_emb[indices]
        pos_emb_ = pos_emb[indices]
        neg_emb_ = neg_emb[indices]
        pos_ = pos.reshape(pos.size)[indices]
        neg_ = neg.reshape(neg.size)[indices]
        
        start_inx = 0
        end_inx = 60
        iterss = 0
        mean_loss = 0
        bpr_loss = 0
        gt_loss = 0
        rc_loss = 0
        clip_rc_loss = 0
        
        while start_inx < len(log_emb_):
            log_emb = log_emb_[start_inx:end_inx]
            pos_emb = pos_emb_[start_inx:end_inx]
            neg_emb = neg_emb_[start_inx:end_inx]
            
            pos__ = pos_[start_inx:end_inx]
            neg__ = neg_[start_inx:end_inx]
            
            start_inx = end_inx
            end_inx += 60
            iterss += 1
            
            # === CHANGED: Get CLIP embeddings instead of SBERT ===
            # Original:
            # pos_text = self.find_item_text(pos__)
            # pos_token = self.sbert.tokenize(pos_text)
            # pos_text_embedding = self.sbert({...})['sentence_embedding']
            
            # New: Direct lookup from pre-computed CLIP embeddings
            pos_clip_embedding = self.get_clip_embedding(pos__)  # [batch, 512]
            neg_clip_embedding = self.get_clip_embedding(neg__)  # [batch, 512]
            
            # Project CF embeddings: 50 -> 128 (latent) -> 50 (recon)
            pos_text_matching, pos_proj = self.mlp(pos_emb)
            neg_text_matching, neg_proj = self.mlp(neg_emb)
            
            # Project CLIP embeddings: 512 -> 128 (latent) -> 512 (recon)
            pos_text_matching_clip, pos_clip_proj = self.mlp2(pos_clip_embedding)
            neg_text_matching_clip, neg_clip_proj = self.mlp2(neg_clip_embedding)
            
            # Recommendation loss (BCE)
            pos_logits = (log_emb * pos_proj).mean(axis=1)
            neg_logits = (log_emb * neg_proj).mean(axis=1)
            pos_labels = torch.ones(pos_logits.shape, device=pos_logits.device)
            neg_labels = torch.zeros(neg_logits.shape, device=neg_logits.device)

            loss = self.bce_criterion(pos_logits, pos_labels)
            loss += self.bce_criterion(neg_logits, neg_labels)
            
            # Matching loss: Align CF latent (128) with CLIP latent (128)
            matching_loss = self.mse(pos_text_matching, pos_text_matching_clip)
            matching_loss += self.mse(neg_text_matching, neg_text_matching_clip)
            
            # CF reconstruction loss: reconstructed (50) vs original (50)
            reconstruction_loss = self.mse(pos_proj, pos_emb) + self.mse(neg_proj, neg_emb)
            
            # CLIP reconstruction loss: reconstructed (512) vs original (512)
            clip_reconstruction_loss = self.mse(pos_clip_proj, pos_clip_embedding.detach())
            clip_reconstruction_loss += self.mse(neg_clip_proj, neg_clip_embedding.detach())
            
            # Total loss (same coefficients as original paper)
            total_loss = loss + matching_loss + 0.5 * reconstruction_loss + 0.2 * clip_reconstruction_loss
            
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            mean_loss += total_loss.item()
            bpr_loss += loss.item()
            gt_loss += matching_loss.item()
            rc_loss += reconstruction_loss.item()
            clip_rc_loss += clip_reconstruction_loss.item()
            
        print("loss in epoch {}/{} iteration {}/{}: {:.4f} / BPR: {:.4f} / Match: {:.4f} / CF-Recon: {:.4f} / CLIP-Recon: {:.4f}".format(
            epoch, total_epoch, step, total_step, 
            mean_loss/iterss, bpr_loss/iterss, gt_loss/iterss, rc_loss/iterss, clip_rc_loss/iterss))
    
    # === Stage 2 methods remain unchanged ===
    # make_interact_text, make_candidate_text, pre_train_phase2, generate
    # These use text for prompts (find_item_text) but embeddings come from get_item_emb
    
    def make_interact_text(self, interact_ids, interact_max_num):
        interact_item_titles_ = self.find_item_text(interact_ids, title_flag=True, description_flag=False)
        interact_text = []
        if interact_max_num == 'all':
            for title in interact_item_titles_:
                interact_text.append(title + '[HistoryEmb]')
        else:
            for title in interact_item_titles_[-interact_max_num:]:
                interact_text.append(title + '[HistoryEmb]')
            interact_ids = interact_ids[-interact_max_num:]
            
        interact_text = ','.join(interact_text)
        return interact_text, interact_ids
    
    def make_candidate_text(self, interact_ids, candidate_num, target_item_id, target_item_title):
        neg_item_id = []
        while len(neg_item_id) < 50:
            t = np.random.randint(1, self.item_num + 1)
            if not (t in interact_ids or t in neg_item_id):
                neg_item_id.append(t)
        random.shuffle(neg_item_id)
        
        candidate_ids = [target_item_id]
        candidate_text = [target_item_title + '[CandidateEmb]']

        for neg_candidate in neg_item_id[:candidate_num - 1]:
            candidate_text.append(self.find_item_text_single(neg_candidate, title_flag=True, description_flag=False) + '[CandidateEmb]')
            candidate_ids.append(neg_candidate)
                
        random_ = np.random.permutation(len(candidate_text))
        candidate_text = np.array(candidate_text)[random_]
        candidate_ids = np.array(candidate_ids)[random_]
            
        return ','.join(candidate_text), candidate_ids
    
    def pre_train_phase2(self, data, optimizer, batch_iter):
        """Stage 2: Train LLM projections - UNCHANGED from original"""
        epoch, total_epoch, step, total_step = batch_iter
        
        optimizer.zero_grad()
        u, seq, pos, neg = data
        mean_loss = 0
        
        text_input = []
        text_output = []
        interact_embs = []
        candidate_embs = []
        self.llm.eval()
        
        with torch.no_grad():
            log_emb = self.recsys.model(u, seq, pos, neg, mode='log_only')
            
        for i in range(len(u)):
            target_item_id = pos[i][-1]
            target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)
            
            interact_text, interact_ids = self.make_interact_text(seq[i][seq[i] > 0], 10)
            candidate_num = 20
            candidate_text, candidate_ids = self.make_candidate_text(seq[i][seq[i] > 0], candidate_num, target_item_id, target_item_title)
            
            input_text = ''
            input_text += ' is a user representation.'
                
            if self.args.rec_pre_trained_data == 'Movies_and_TV':
                input_text += 'This user has watched '
            elif self.args.rec_pre_trained_data == 'Video_Games':
                input_text += 'This user has played '
            elif self.args.rec_pre_trained_data == 'Luxury_Beauty' or self.args.rec_pre_trained_data == 'Toys_and_Games' or self.args.rec_pre_trained_data == 'All_Beauty':
                input_text += 'This user has bought '
                
            input_text += interact_text
            
            if self.args.rec_pre_trained_data == 'Movies_and_TV':
                input_text += ' in the previous. Recommend one next movie for this user to watch next from the following movie title set, '
            elif self.args.rec_pre_trained_data == 'Video_Games':
                input_text += ' in the previous. Recommend one next game for this user to play next from the following game title set, '            
            elif self.args.rec_pre_trained_data == 'Luxury_Beauty' or self.args.rec_pre_trained_data == 'Toys_and_Games' or self.args.rec_pre_trained_data == 'All_Beauty':
                input_text += ' in the previous. Recommend one next item for this user to buy next from the following item title set, '
                    
            input_text += candidate_text
            input_text += '. The recommendation is '

            text_input.append(input_text)
            text_output.append(target_item_title)

            interact_embs.append(self.item_emb_proj(self.get_item_emb(interact_ids)))
            candidate_embs.append(self.item_emb_proj(self.get_item_emb(candidate_ids)))
            
        samples = {'text_input': text_input, 'text_output': text_output, 'interact': interact_embs, 'candidate': candidate_embs}
        log_emb = self.log_emb_proj(log_emb)
        loss_rm = self.llm(log_emb, samples)
        loss_rm.backward()
        optimizer.step()
        mean_loss += loss_rm.item()
        print("A-LLMRec model loss in epoch {}/{} iteration {}/{}: {}".format(epoch, total_epoch, step, total_step, mean_loss))
        
    def generate(self, data):
        """Generate recommendations - UNCHANGED from original"""
        u, seq, pos, neg, rank = data
        
        answer = []
        text_input = []
        interact_embs = []
        candidate_embs = []
        
        with torch.no_grad():
            log_emb = self.recsys.model(u, seq, pos, neg, mode='log_only')
            for i in range(len(u)):
                target_item_id = pos[i]
                target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)
                
                interact_text, interact_ids = self.make_interact_text(seq[i][seq[i] > 0], 10)
                candidate_num = 20
                candidate_text, candidate_ids = self.make_candidate_text(seq[i][seq[i] > 0], candidate_num, target_item_id, target_item_title)
                
                input_text = ''
                input_text += ' is a user representation.'
                if self.args.rec_pre_trained_data == 'Movies_and_TV':
                    input_text += 'This user has watched '
                elif self.args.rec_pre_trained_data == 'Video_Games':
                    input_text += 'This user has played '
                elif self.args.rec_pre_trained_data == 'Luxury_Beauty' or self.args.rec_pre_trained_data == 'Toys_and_Games' or self.args.rec_pre_trained_data == 'All_Beauty':
                    input_text += 'This user has bought '
                    
                input_text += interact_text
                
                if self.args.rec_pre_trained_data == 'Movies_and_TV':
                    input_text += ' in the previous. Recommend one next movie for this user to watch next from the following movie title set, '
                elif self.args.rec_pre_trained_data == 'Video_Games':
                    input_text += ' in the previous. Recommend one next game for this user to play next from the following game title set, '            
                elif self.args.rec_pre_trained_data == 'Luxury_Beauty' or self.args.rec_pre_trained_data == 'Toys_and_Games' or self.args.rec_pre_trained_data == 'All_Beauty':
                    input_text += ' in the previous. Recommend one next item for this user to buy next from the following item title set, '
                
                input_text += candidate_text
                input_text += '. The recommendation is '
                
                answer.append(target_item_title)
                text_input.append(input_text)
                
                interact_embs.append(self.item_emb_proj(self.get_item_emb(interact_ids)))
                candidate_embs.append(self.item_emb_proj(self.get_item_emb(candidate_ids)))
        
        log_emb = self.log_emb_proj(log_emb)
        atts_llm = torch.ones(log_emb.size()[:-1], dtype=torch.long).to(self.device)
        atts_llm = atts_llm.unsqueeze(1)
        log_emb = log_emb.unsqueeze(1)
        
        with torch.no_grad():
            self.llm.llm_tokenizer.padding_side = "left"
            llm_tokens = self.llm.llm_tokenizer(
                text_input,
                padding="longest",
                return_tensors="pt"
            ).to(self.device)
            
            with torch.cuda.amp.autocast():
                inputs_embeds = self.llm.llm_model.get_input_embeddings()(llm_tokens.input_ids)
                llm_tokens, inputs_embeds = self.llm.replace_hist_candi_token(llm_tokens, inputs_embeds, interact_embs, candidate_embs)
                    
                attention_mask = llm_tokens.attention_mask
                inputs_embeds = torch.cat([log_emb, inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)
                
                outputs = self.llm.llm_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=256,
                    min_new_tokens=1,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.llm.llm_tokenizer.pad_token_id,
                    eos_token_id=self.llm.llm_tokenizer.eos_token_id,
                    repetition_penalty=1.5,
                    length_penalty=1.0,
                    num_return_sequences=1,
                )

            outputs[outputs == 0] = 2
            output_text = self.llm.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text = [text.strip() for text in output_text]

        for i in range(len(text_input)):
            f = open(f'./recommendation_output_smol_clip_69.txt', 'a')
            f.write(text_input[i])
            f.write('\n\n')
            f.write('Answer: ' + answer[i])
            f.write('\n\n')
            f.write('LLM: ' + str(output_text[i]))
            f.write('\n\n')
            f.close()

        return output_text