import numpy as np
from copy import copy
import torch
import torch.nn.functional as F
from utils.utils import get_loss
from utils.utils import get_top_p_indices

class BeamConfig:
    def __init__(self, llm, llm_tokenizer, embedding_model, emb_tokenizer, target_sentence, seq_len, device):
        
        self.llm = llm
        self.embedding_model = embedding_model
        self.llm_tokenizer = llm_tokenizer
        self.emb_tokenizer = emb_tokenizer

        self.device = device

        text = f"Represent this sentence for searching relevant passages: {target_sentence}"
        inputs = emb_tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = self.embedding_model(**inputs)

        # Mean pooling to get sentence embedding
        TARGET = outputs.last_hidden_state.mean(dim=1)  # shape: (1, hidden_size)
        TARGET = F.normalize(TARGET, p=2.0, dim=-1).to(device)
        self.target = TARGET

        self.seq_len = seq_len

class Beam:
    def __init__(self, bc, text=""):
        self.text = text
        self.llm_tokenizer = bc.llm_tokenizer
        self.emb_tokenizer = bc.emb_tokenizer
        self.len = 0 # len for the purpose of generating probability scores
        self.sum_prob = 0.
        self.avg_prob = 0.
        self.similarity_score = 0

    def update_similarity(self, bc):
        tokens = self.emb_tokenizer(
            self.text,
            add_special_tokens=True
            )["input_ids"]
        loss = get_loss(
            torch.tensor(tokens),
            bc)
        self.similarity_score = 1-loss
        return 1-loss

    def update_text(self, text, prob):
        self.text = text
        self.sum_prob += prob
        self.len += 1
        self.sum_prob = self.sum_prob / self.len


class Beams:
    def __init__(self, beam_size, text, bc,
                 scoring_function={"prob_score": 1.,
                                "similarity_score": 5.,
                                "len_score": 1/128}
                 ):

        self.beams = [Beam(bc, text) for i in range(beam_size)]

        self.beam_size = beam_size
        self.scoring_function = scoring_function
        self.best = {"beam" : None, "score" : 0}
        self.beam_config = bc

    def update_similarities(self, idx=None):
        if idx == None:
            for beam in self.beams:
                beam.update_similarity(self.beam_config)
        else:
            self.beams[idx].update_similarity(self.beam_config)
    
    def score(self, beam, mini=0, maxi=1):
        """Minimax standardization plus scaling"""
        prob_score = beam.avg_prob
        prob_score = ((prob_score - mini) / 
                        (maxi - mini))

        return (beam.similarity_score * self.scoring_function["similarity_score"] +
                    prob_score * self.scoring_function["prob_score"] + 
                    beam.len * self.scoring_function["len_score"])
    
    def update_best(self):
        for beam in self.beams:
            score = self.score(beam)
            if score > self.best["score"]:
                self.best["beam"] = beam
                self.best["score"] = score

    def cull(self):
        k = min(
            len(self.beams), 
            self.beam_size
            )        
              
        scores = torch.tensor([self.score(beam) for beam in self.beams])
        top_k = torch.topk(scores, k).indices

        self.beams = [
            self.beams[i] for i in range(len(self.beams))
            if i in top_k
            ]
        
    def get_softmax(self, token_list, top_p, temp):
        tokens = torch.tensor([token_list])
        assert len(tokens.shape) == 2
        with torch.no_grad():
            activations = self.beam_config.llm(tokens)
            sm_inputs = torch.squeeze(activations.logits[:,-1, :])
            sm = torch.softmax(sm_inputs, dim=-1).to('cpu') #tok_llm = tok_llm-2 = tok_emb-4
            
            not_top_p_indices, top_p_indices = get_top_p_indices(sm, top_p) 
            sm[not_top_p_indices] = float('-inf')
            sm = torch.softmax(sm / temp, dim=-1)

        return sm, len(top_p_indices)
    
    def search(self, top_p=0.7, temp=1.5):
        new_beams = []
        for beam in self.beams:
            tokens = beam.llm_tokenizer(
                        beam.text, 
                        add_special_tokens=True,
                        )["input_ids"]
            assert beam.llm_tokenizer.all_special_ids
            assert tokens[-1] not in beam.llm_tokenizer.all_special_ids
            #TODO: put tokens in pt format
            sm, k = self.get_softmax(tokens, top_p, temp)
            k = min(self.beam_size, k)
            for token in tokens[-10:]:
                sm[token] /= 2.0

            new_tokens = torch.multinomial(sm, k)
            new_tokens_probs = sm[new_tokens]

            for token, prob in zip(new_tokens, new_tokens_probs):
                new_beam = copy(beam)
                beam_toks = tokens + [token]
                beam_text = new_beam.llm_tokenizer.decode(beam_toks, skip_special_tokens=True)
                new_beam.update_text(beam_text, prob)
                new_beams.append(new_beam)
                
        self.beams = new_beams
        
