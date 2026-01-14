# llm4rec.py
# SmolVLM2-2.2B-Instruct integration for A-LLMRec with CLIP embeddings
# Fixed version with proper dtype handling for flash attention

import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForImageTextToText


class llm4rec(nn.Module):
    """
    SmolVLM2 wrapper for recommendation task.
    
    SmolVLM2 Architecture (from paper):
    - Vision Encoder: SigLIP-SO400M (400M params)
    - Language Model: SmolLM2-1.7B 
    - Total: 2.2B parameters
    - Hidden size: 2048
    - Context length: 16k tokens
    """
    
    def __init__(
        self,
        device,
        model_path="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        max_output_txt_len=256,
    ):
        super().__init__()
        self.device = device
        self.max_output_txt_len = max_output_txt_len
        self.model_path = model_path
        
        print("\n" + "="*70)
        print("INITIALIZING SmolVLM2 FOR RECOMMENDATION")
        print("="*70)
        print(f"[SmolVLM2] Model: {model_path}")
        print(f"[SmolVLM2] Device: {device}")
        
        # Load processor
        print("[SmolVLM2] Loading processor...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        # Load model with bfloat16
        print("[SmolVLM2] Loading model (this may take a moment)...")
        self.llm_model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            _attn_implementation="flash_attention_2"
        )
        
        # Store the model dtype for consistency
        self.model_dtype = torch.bfloat16
        
        # Enable memory-efficient training
        self.llm_model.gradient_checkpointing_enable()
        self.llm_model.config.use_cache = False
        
        # Get tokenizer
        self.llm_tokenizer = self.processor.tokenizer
        
        print(f"[SmolVLM2] Tokenizer vocab size: {len(self.llm_tokenizer)}")
        print(f"[SmolVLM2] BOS token: {self.llm_tokenizer.bos_token} (ID: {self.llm_tokenizer.bos_token_id})")
        print(f"[SmolVLM2] EOS token: {self.llm_tokenizer.eos_token} (ID: {self.llm_tokenizer.eos_token_id})")
        print(f"[SmolVLM2] PAD token: {self.llm_tokenizer.pad_token} (ID: {self.llm_tokenizer.pad_token_id})")
        
        # Add custom tokens for recommendation
        self.rec_special_tokens = ['[UserRep]', '[HistoryEmb]', '[CandidateEmb]']
        
        existing_tokens = set(self.llm_tokenizer.get_vocab().keys())
        tokens_to_add = [t for t in self.rec_special_tokens if t not in existing_tokens]
        
        if tokens_to_add:
            num_added = self.llm_tokenizer.add_special_tokens({
                'additional_special_tokens': tokens_to_add
            })
            print(f"[SmolVLM2] Added {num_added} recommendation tokens: {tokens_to_add}")
            self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
            print(f"[SmolVLM2] Resized embeddings to: {len(self.llm_tokenizer)}")
        else:
            print(f"[SmolVLM2] Recommendation tokens already exist")
        
        # Store special token IDs
        self.userrep_token_id = self.llm_tokenizer.convert_tokens_to_ids("[UserRep]")
        self.history_token_id = self.llm_tokenizer.convert_tokens_to_ids("[HistoryEmb]")
        self.candidate_token_id = self.llm_tokenizer.convert_tokens_to_ids("[CandidateEmb]")
        
        print(f"[SmolVLM2] Recommendation token IDs:")
        print(f"          [UserRep]: {self.userrep_token_id}")
        print(f"          [HistoryEmb]: {self.history_token_id}")
        print(f"          [CandidateEmb]: {self.candidate_token_id}")
        
        # Verify tokens are not UNK
        unk_id = self.llm_tokenizer.unk_token_id
        if any(tid == unk_id for tid in [self.userrep_token_id, self.history_token_id, self.candidate_token_id]):
            print("[SmolVLM2] WARNING: Some recommendation tokens mapped to UNK!")
        
        # Freeze base model
        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False
        print("[SmolVLM2] Froze all model parameters")
        
        # Get hidden size
        self.hidden_size = self._get_hidden_size()
        print(f"[SmolVLM2] Hidden size: {self.hidden_size}")
        
        print("[SmolVLM2] Initialization complete!")
        print("="*70 + "\n")

    def _get_hidden_size(self):
        """Extract hidden size from SmolVLM2 config"""
        config = self.llm_model.config
        
        if hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
            return config.text_config.hidden_size
        if hasattr(config, 'hidden_size'):
            return config.hidden_size
        
        emb = self.llm_model.get_input_embeddings()
        if hasattr(emb, 'embedding_dim'):
            return emb.embedding_dim
        
        print("[SmolVLM2] WARNING: Could not determine hidden size, using default 2048")
        return 2048

    def tokenize_text(self, text_list, add_special_tokens=True, max_length=2048):
        """Tokenize text for SmolVLM2"""
        self.llm_tokenizer.padding_side = "right"
        
        tokens = self.llm_tokenizer(
            text_list,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length,
            add_special_tokens=add_special_tokens
        ).to(self.device)
        
        return tokens

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        """Concatenate input and output for teacher-forcing training"""
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum().item()
            input_part_targets_len.append(int(this_input_ones))
            
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        
        return llm_tokens, input_part_targets_len

    def replace_special_tokens(self, llm_tokens, inputs_embeds, user_reps=None, 
                                interact_embs=None, candidate_embs=None):
        """
        Replace recommendation special tokens with actual embeddings.
        
        IMPORTANT: All embeddings must be in bfloat16 for flash attention compatibility!
        """
        batch_size = inputs_embeds.size(0)
        target_dtype = inputs_embeds.dtype  # Should be bfloat16
        
        for idx in range(batch_size):
            input_ids = llm_tokens["input_ids"][idx]
            
            # Replace [UserRep] token
            if user_reps is not None:
                userrep_positions = (input_ids == self.userrep_token_id).nonzero(as_tuple=False).view(-1)
                if len(userrep_positions) > 0:
                    # Ensure dtype matches
                    user_emb = user_reps[idx].to(dtype=target_dtype)
                    inputs_embeds[idx, userrep_positions[0]] = user_emb
            
            # Replace [HistoryEmb] tokens
            if interact_embs is not None and idx < len(interact_embs):
                hist_embs = interact_embs[idx]
                if hist_embs is not None and len(hist_embs) > 0:
                    hist_positions = (input_ids == self.history_token_id).nonzero(as_tuple=False).view(-1)
                    num_replace = min(len(hist_positions), len(hist_embs))
                    for pos_idx in range(num_replace):
                        # Ensure dtype matches
                        emb = hist_embs[pos_idx].to(dtype=target_dtype)
                        inputs_embeds[idx, hist_positions[pos_idx]] = emb
            
            # Replace [CandidateEmb] tokens
            if candidate_embs is not None and idx < len(candidate_embs):
                cand_embs = candidate_embs[idx]
                if cand_embs is not None and len(cand_embs) > 0:
                    cand_positions = (input_ids == self.candidate_token_id).nonzero(as_tuple=False).view(-1)
                    num_replace = min(len(cand_positions), len(cand_embs))
                    for pos_idx in range(num_replace):
                        # Ensure dtype matches
                        emb = cand_embs[pos_idx].to(dtype=target_dtype)
                        inputs_embeds[idx, cand_positions[pos_idx]] = emb
        
        return inputs_embeds

    def replace_hist_candi_token(self, llm_tokens, inputs_embeds, interact_embs, candidate_embs):
        """Backward compatibility wrapper"""
        return llm_tokens, self.replace_special_tokens(
            llm_tokens, inputs_embeds, 
            user_reps=None, 
            interact_embs=interact_embs, 
            candidate_embs=candidate_embs
        )

    def forward(self, log_emb, samples):
        """
        Training forward pass.
        
        Args:
            log_emb: User representation [batch, hidden_size] - should be in bfloat16
            samples: Dict with text_input, text_output, interact, candidate
        
        Returns:
            loss: Scalar loss value
        """
        batch_size = log_emb.size(0)
        
        # Ensure log_emb is bfloat16
        log_emb = log_emb.to(dtype=self.model_dtype)
        
        # Create attention for prepended user representation
        atts_llm = torch.ones((batch_size, 1), dtype=torch.long, device=self.device)
        
        # Tokenize outputs with EOS
        text_output_with_eos = [t + self.llm_tokenizer.eos_token for t in samples['text_output']]
        output_tokens = self.llm_tokenizer(
            text_output_with_eos,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
            add_special_tokens=False
        ).to(self.device)
        
        # Tokenize inputs
        input_tokens = self.tokenize_text(samples['text_input'], add_special_tokens=True)
        
        # Concatenate for teacher forcing
        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            input_tokens['input_ids'],
            input_tokens['attention_mask'],
            output_tokens['input_ids'],
            output_tokens['attention_mask'],
        )
        
        # Create targets
        targets = llm_tokens['input_ids'].clone()
        targets[llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id] = -100
        
        for i, l in enumerate(input_part_targets_len):
            targets[i, :l] = -100
        
        empty_targets = torch.full((batch_size, 1), -100, dtype=torch.long, device=self.device)
        targets = torch.cat([empty_targets, targets], dim=1)
        
        # Get token embeddings (will be in bfloat16)
        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
        
        # Replace special tokens - embeddings will be cast to bfloat16 inside
        inputs_embeds = self.replace_special_tokens(
            llm_tokens,
            inputs_embeds,
            user_reps=log_emb,
            interact_embs=samples.get('interact'),
            candidate_embs=samples.get('candidate')
        )
        
        # Prepend user representation (ensure bfloat16)
        log_emb_expanded = log_emb.unsqueeze(1).to(dtype=inputs_embeds.dtype)
        inputs_embeds = torch.cat([log_emb_expanded, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)
        
        # Forward pass
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        
        return outputs.loss

    @torch.no_grad()
    def generate(self, log_emb, samples, max_new_tokens=64):
        """
        Generate recommendations.
        
        CRITICAL: All tensors must be bfloat16 for flash attention!
        """
        batch_size = log_emb.size(0)
        
        # === ENSURE BFLOAT16 FOR FLASH ATTENTION ===
        log_emb = log_emb.to(dtype=self.model_dtype)
        
        # Attention for user rep
        atts_llm = torch.ones((batch_size, 1), dtype=torch.long, device=self.device)
        
        # Tokenize input with left padding for generation
        self.llm_tokenizer.padding_side = "left"
        input_tokens = self.llm_tokenizer(
            samples['text_input'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=2048,
            add_special_tokens=True
        ).to(self.device)
        
        # Get embeddings (already in bfloat16 from model)
        inputs_embeds = self.llm_model.get_input_embeddings()(input_tokens['input_ids'])
        
        # Convert interact and candidate embeddings to bfloat16
        if samples.get('interact') is not None:
            interact_embs = []
            for embs in samples['interact']:
                if embs is not None:
                    interact_embs.append(embs.to(dtype=self.model_dtype))
                else:
                    interact_embs.append(None)
        else:
            interact_embs = None
            
        if samples.get('candidate') is not None:
            candidate_embs = []
            for embs in samples['candidate']:
                if embs is not None:
                    candidate_embs.append(embs.to(dtype=self.model_dtype))
                else:
                    candidate_embs.append(None)
        else:
            candidate_embs = None
        
        # Replace special tokens (will ensure bfloat16)
        inputs_embeds = self.replace_special_tokens(
            input_tokens,
            inputs_embeds,
            user_reps=log_emb,
            interact_embs=interact_embs,
            candidate_embs=candidate_embs
        )
        
        # Prepend user rep (ensure bfloat16)
        log_emb_expanded = log_emb.unsqueeze(1).to(dtype=inputs_embeds.dtype)
        inputs_embeds = torch.cat([log_emb_expanded, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, input_tokens['attention_mask']], dim=1)
        
        # === ENSURE ALL TENSORS ARE BFLOAT16 ===
        inputs_embeds = inputs_embeds.to(dtype=self.model_dtype)
        
        # Disable cache for generation (important for flash attention)
        self.llm_model.config.use_cache = True  # Need cache for generation
        
        # Generate
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.llm_tokenizer.pad_token_id,
                eos_token_id=self.llm_tokenizer.eos_token_id,
                repetition_penalty=1.5,
                use_cache=True,  # Enable for generation
            )
        
        # Reset cache setting
        self.llm_model.config.use_cache = False
        
        # Decode
        generated_texts = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generated_texts = [text.strip() for text in generated_texts]
        
        return generated_texts