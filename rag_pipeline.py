import io
import re
import logging
import argparse
from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

from config import Config

logging.basicConfig(level=logging.INFO)


def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool token embeddings with attention mask.
    last_hidden_state: [B, T, H], attention_mask: [B, T]
    returns: [B, H]
    """
    mask = attention_mask.unsqueeze(-1)  # [B, T, 1]
    summed = (last_hidden_state * mask).sum(dim=1)            # [B, H]
    counts = mask.sum(dim=1).clamp(min=1e-9)                  # [B, 1]
    return summed / counts


class RAGPipeline:
    def __init__(
        self,
        db_path: str = Config.db_path,
        emb_model: str = Config.embedding_model,
        reranker: str = Config.reranker,
        generator_model: str = Config.generator_model,
        max_chunk_tokens: int = 64,
        top_k_retrieval: int = 20,
        top_k_rerank: int = 5,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.db_path = db_path

        # Embedding tokenizer and model
        self.emb_tokenizer = AutoTokenizer.from_pretrained(emb_model, use_fast=True)
        self.emb_model_name = emb_model

        self.embedding_model = AutoModel.from_pretrained(self.emb_model_name)
        self.embedding_model.eval()
        self.embedding_model.to(self.device)

        # self.chunks = self._chunk_text_by_tokens(raw_text, self.emb_tokenizer, max_chunk_tokens) # Chunk text based on 

        # Reranker
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker)
        self.reranker_model.eval()
        self.reranker_model.to(self.device)

        # Generator LLM + tokenizer
        self.gen_tokenizer = AutoTokenizer.from_pretrained(generator_model, use_fast=True)
        # Some chat/instruct models need a pad token set for generate()
        if self.gen_tokenizer.pad_token_id is None and self.gen_tokenizer.eos_token_id is not None:
            self.gen_tokenizer.pad_token = self.gen_tokenizer.eos_token
        self.generator_model = AutoModelForCausalLM.from_pretrained(generator_model)
        self.generator_model.eval()
        self.generator_model.to(self.device)

        # Vector db access
        client = chromadb.PersistentClient(path=self.db_path)
        huggingface_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.emb_model_name)
        self.collection = client.get_collection(
            name=Config.collection_name,
            embedding_function=huggingface_ef, 
        )

        # Config
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank

    # Helper
    @staticmethod
    def _chunk_text_by_tokens(text: str, tokenizer: AutoTokenizer, max_tokens: int) -> List[str]:
        '''
        Sentence-wise chunking for the cat document. For wikipedia, we will use another method, possibly semantic chunking.
        '''
        pieces = [p.strip() for p in text.split("\n") if p.strip()]
        return pieces
    

    def _embed_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Return masked-mean sentence embeddings for a batch of texts. Shape: [B, H]
        """
        enc = self.emb_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            out = self.embedding_model(**enc)
            sent = masked_mean_pool(out.last_hidden_state, enc.attention_mask)
            sent = F.normalize(sent, p=2, dim=1)
        return sent

    def _retrieve(self, query: str) -> List[Tuple[str, float]]:
        """
        Cosine-sim retrieval using embedding model. Returns top_k_retrieval [(chunk, score)].
        """
        logging.info("Retrieving documents...")
        retrieved = self.collection.query(query_texts=query) # Sort this based on 'distances'
        docs = retrieved["documents"][0]     # list of retrieved texts
        scores = retrieved["distances"][0]   # list of distances (lower = closer)

        pairs = list(zip(docs, scores))

        pairs.sort(key=lambda x: x[1])
        return pairs[: self.top_k_retrieval]


    def _rerank(self, query: str, candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Cross-encoder rerank with HF sequence classification model. Returns top_k_rerank [(chunk, score)].
        """
        logging.info("Reranking...")
        texts_a = [query] * len(candidates)
        texts_b = [c[0] for c in candidates]

        enc = self.reranker_tokenizer(
            texts_a,
            texts_b,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.reranker_model(**enc).logits.squeeze(-1)  # [B] or [B,1]->[B]
            # Many rerankers are trained with a regression head; we can use raw logits or sigmoid
            scores = logits.tolist()

        reranked = list(zip(texts_b, scores))
        reranked.sort(key=lambda x: x[1], reverse=True) 
        return reranked[: self.top_k_rerank]

    def _build_prompt(self, query: str, top_docs: List[Tuple[str, float]]) -> str:
        """
        Simple prompt: user question + concatenated top docs.
        """
        context = "\n\n".join([d for d, _ in top_docs])
        return (
            "You are a helpful assistant that answers using the provided context.\n\n"
            f"Question:\n{query}\n\n"
            f"Context:\n{context}\n\n"
            "Answer concisely and cite relevant lines from the context. Do not make things up. Only use the information provided in the context. Make sure your answer is relevant to the question provided above, and do not provide any additional information whatsoever. If information is not provided in the context, say you do not know."
        )

    def _generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        enc = self.gen_tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = enc.input_ids.shape[1]  # number of tokens in the prompt
        with torch.no_grad():
            out = self.generator_model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                pad_token_id=self.gen_tokenizer.pad_token_id,
                eos_token_id=self.gen_tokenizer.eos_token_id,
            )
        # Slice out only the newly generated tokens
        generated_tokens = out[0][input_len:]
        return self.gen_tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    def poisoned_test(self, prompt: str, url: str):
        url_format = r"[a-zA-Z0-9-]+(\.[a-zA-Z]{2,})" # xyz.com - TLD should be 2 letters or more. 
        cmp = re.compile(url_format)

        generated_text = self.get_response(prompt)
        # print("Generated text: ", generated_text)

        for word in generated_text.split():
            if mtch:=cmp.match(word): # check if any url exists
                match_flag = (mtch.group() == url) # check if the specific url passed from cmdline exists
                # print("Match", match_flag)
                if match_flag:
                    return match_flag
            else:
                return False


    def get_response(self, query: str) -> str:
        retrieved = self._retrieve(query)
        reranked = self._rerank(query, retrieved)
        prompt = self._build_prompt(query, reranked)
        print("----- Prompt: ", prompt, "\n-----\n\n")
        return self._generate(prompt)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Query and URL to look for")
    parser.add_argument("todo", choices=["generate", "test"])
    parser.add_argument("-q", "--query")
    parser.add_argument("-u", "--url")

    args = parser.parse_args()
    # print(args.todo, args.query, args.url, type(args.todo), type(args.query), type(args.url))

    pipeline = RAGPipeline()
    if args.todo=="generate":
        print(pipeline.get_response(args.query))
    else:
        if args.url:
            print(pipeline.poisoned_test(args.query, args.url))
        else:
            raise TypeError("pass url as argument to check for poisoning using the -u flag")
