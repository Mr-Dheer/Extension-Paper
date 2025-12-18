import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForImageTextToText

class llm4rec(nn.Module):
    def __init__(
        self,
        device,
        max_output_txt_len=256,
    ):
        super().__init__()
        self.device = device
        
        # Load SmolVLM2 model and processor
        model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
        
        print(f"Loading SmolVLM2 from {model_path}...")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        # Load model without quantization
        self.llm_model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            _attn_implementation="flash_attention_2"
        )
        
        self.llm_model.gradient_checkpointing_enable()
        self.llm_model.config.use_cache = False
        
        # Access the underlying tokenizer
        self.llm_tokenizer = self.processor.tokenizer
        
        print(f"Original vocab size: {len(self.llm_tokenizer)}")
        
        # Add special tokens for recommendation task
        special_tokens_dict = {
            'additional_special_tokens': ['[UserRep]', '[HistoryEmb]', '[CandidateEmb]']
        }
        
        num_added_tokens = self.llm_tokenizer.add_special_tokens(special_tokens_dict)
        print(f"Added {num_added_tokens} special tokens")
        
        # Resize token embeddings if new tokens were added
        if num_added_tokens > 0:
            self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
            print(f"New vocab size: {len(self.llm_tokenizer)}")
        
        # Freeze base model parameters
        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False
        
        self.max_output_txt_len = max_output_txt_len
        
        # Store special token IDs for efficient lookup
        self.userrep_token_id = self.llm_tokenizer.convert_tokens_to_ids("[UserRep]")
        self.history_token_id = self.llm_tokenizer.convert_tokens_to_ids("[HistoryEmb]")
        self.candidate_token_id = self.llm_tokenizer.convert_tokens_to_ids("[CandidateEmb]")
        
        print(f"Special token IDs - UserRep: {self.userrep_token_id}, "
              f"HistoryEmb: {self.history_token_id}, CandidateEmb: {self.candidate_token_id}")
        
        print("SmolVLM2 loaded successfully!")

    def tokenize_text_simple(self, text_list):
        """
        Simple tokenization without chat template - used for custom prompts
        """
        # Tokenize directly without chat template
        tokens = self.llm_tokenizer(
            text_list,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            add_special_tokens=True  # Add BOS/EOS
        ).to(self.device)
        
        return tokens

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        """Concatenate input and output sequences for teacher forcing"""
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],  # Skip BOS token from output
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

    def replace_hist_candi_token(self, llm_tokens, inputs_embeds, interact_embs, candidate_embs):
        """
        Replace history and candidate special tokens with item embeddings.
        This method is called from generate() in a_llmrec_model.py
        
        Args:
            llm_tokens: Dict with 'input_ids' and 'attention_mask'
            inputs_embeds: Token embeddings tensor [batch_size, seq_len, hidden_dim]
            interact_embs: List of interaction history embeddings for each batch item
            candidate_embs: List of candidate item embeddings for each batch item
        
        Returns:
            llm_tokens: Unchanged token dict
            inputs_embeds: Updated embeddings with special tokens replaced
        """
        batch_size = len(llm_tokens["input_ids"])
        
        for idx in range(batch_size):
            # Replace [HistoryEmb] tokens with interaction embeddings
            if interact_embs is not None and len(interact_embs[idx]) > 0:
                hist_positions = (llm_tokens["input_ids"][idx] == self.history_token_id).nonzero(as_tuple=False).view(-1)
                for pos, item_emb in zip(hist_positions, interact_embs[idx]):
                    inputs_embeds[idx][pos] = item_emb
            
            # Replace [CandidateEmb] tokens with candidate embeddings
            if candidate_embs is not None and len(candidate_embs[idx]) > 0:
                cand_positions = (llm_tokens["input_ids"][idx] == self.candidate_token_id).nonzero(as_tuple=False).view(-1)
                for pos, item_emb in zip(cand_positions, candidate_embs[idx]):
                    inputs_embeds[idx][pos] = item_emb
        
        return llm_tokens, inputs_embeds

    def replace_special_tokens(self, llm_tokens, inputs_embeds, user_reps, interact_embs, candidate_embs):
        """
        Replace ALL special tokens with actual embeddings.
        This method is used during training (forward pass).
        
        Args:
            llm_tokens: Dict with 'input_ids' and 'attention_mask'
            inputs_embeds: Token embeddings tensor
            user_reps: User representation embeddings [batch_size, hidden_dim]
            interact_embs: List of interaction history embeddings
            candidate_embs: List of candidate item embeddings
        """
        batch_size = len(llm_tokens["input_ids"])
        
        for idx in range(batch_size):
            # Replace [UserRep] with user representation
            userrep_positions = (llm_tokens["input_ids"][idx] == self.userrep_token_id).nonzero(as_tuple=False).view(-1)
            if len(userrep_positions) > 0 and user_reps is not None:
                inputs_embeds[idx][userrep_positions[0]] = user_reps[idx]
            
            # Replace [HistoryEmb] tokens with interaction embeddings
            if interact_embs is not None and len(interact_embs[idx]) > 0:
                hist_positions = (llm_tokens["input_ids"][idx] == self.history_token_id).nonzero(as_tuple=False).view(-1)
                for pos, item_emb in zip(hist_positions, interact_embs[idx]):
                    inputs_embeds[idx][pos] = item_emb
            
            # Replace [CandidateEmb] tokens with candidate embeddings
            if candidate_embs is not None and len(candidate_embs[idx]) > 0:
                cand_positions = (llm_tokens["input_ids"][idx] == self.candidate_token_id).nonzero(as_tuple=False).view(-1)
                for pos, item_emb in zip(cand_positions, candidate_embs[idx]):
                    inputs_embeds[idx][pos] = item_emb
        
        return llm_tokens, inputs_embeds
    
    def forward(self, log_emb, samples):
        """
        Forward pass for recommendation task
        
        Args:
            log_emb: User representation embeddings [batch_size, hidden_dim]
            samples: Dict containing:
                - text_input: List of input prompts (with special tokens)
                - text_output: List of target outputs
                - interact: List of interaction history embeddings
                - candidate: List of candidate item embeddings
        """
        # Create attention mask for user representation
        atts_llm = torch.ones(log_emb.size()[:-1], dtype=torch.long).to(self.device)
        atts_llm = atts_llm.unsqueeze(1)  # [batch_size, 1]
        
        # Tokenize output (targets) - simple tokenization
        text_output_with_eos = [t + self.llm_tokenizer.eos_token for t in samples['text_output']]
        text_output_tokens = self.llm_tokenizer(
            text_output_with_eos,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
            add_special_tokens=False
        ).to(self.device)
        
        # Tokenize input (prompts) - simple tokenization
        text_input_tokens = self.tokenize_text_simple(samples['text_input'])
        
        # Concatenate input and output for teacher forcing
        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens['input_ids'],
            text_input_tokens['attention_mask'],
            text_output_tokens['input_ids'],
            text_output_tokens['attention_mask'],
        )
        
        # Create targets (mask input part, only compute loss on output)
        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, 
            -100
        )
        
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100  # Don't compute loss on input tokens
        
        # Create empty targets for user representation
        empty_targets = (
            torch.ones(atts_llm.size(), dtype=torch.long)
            .to(self.device)
            .fill_(-100)
        )
        
        # Concatenate user rep targets with text targets
        targets = torch.cat([empty_targets, targets], dim=1)
        
        # Get text embeddings from the model
        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
        
        # Replace special tokens with actual embeddings
        llm_tokens, inputs_embeds = self.replace_special_tokens(
            llm_tokens, 
            inputs_embeds,
            log_emb,  # user representations
            samples['interact'],  # history embeddings
            samples['candidate']  # candidate embeddings
        )
        
        # Prepend user representation embedding
        log_emb = log_emb.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        inputs_embeds = torch.cat([log_emb, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)
        
        # Forward pass through model
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        
        loss = outputs.loss
        return loss
    
    @torch.no_grad()
    def generate(self, log_emb, samples, max_new_tokens=64):
        """
        Generate predictions for recommendation
        
        Args:
            log_emb: User representation embeddings
            samples: Dict with text_input, interact, candidate
            max_new_tokens: Maximum tokens to generate
        """
        atts_llm = torch.ones(log_emb.size()[:-1], dtype=torch.long).to(self.device)
        atts_llm = atts_llm.unsqueeze(1)
        
        # Tokenize input
        text_input_tokens = self.tokenize_text_simple(samples['text_input'])
        
        inputs_embeds = self.llm_model.get_input_embeddings()(text_input_tokens['input_ids'])
        
        # Replace special tokens
        _, inputs_embeds = self.replace_special_tokens(
            text_input_tokens,
            inputs_embeds,
            log_emb,
            samples['interact'],
            samples['candidate']
        )
        
        # Prepend user representation
        log_emb = log_emb.unsqueeze(1)
        inputs_embeds = torch.cat([log_emb, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, text_input_tokens['attention_mask']], dim=1)
        
        # Generate
        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.llm_tokenizer.pad_token_id,
            eos_token_id=self.llm_tokenizer.eos_token_id,
        )
        
        generated_texts = self.processor.batch_decode(
            outputs,
            skip_special_tokens=True
        )
        
        return generated_texts