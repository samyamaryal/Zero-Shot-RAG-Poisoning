from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import chromadb
from uuid import uuid4
import chromadb.utils.embedding_functions as embedding_functions
from typing import List, Dict, Any, Tuple

from config import Config


class WikiChromaIndexer:
    def __init__(
        self,
        dataset_name: str = Config.dataset_name,
        subset_name: str = Config.subset_name,
        chroma_path: str = Config.db_name,
        collection_name: str = Config.collection_name,
        embedding_model_name: str = Config.embedding_model,
        batch_size: int = 500,
    ):
        self.dataset_name = dataset_name
        self.subset_name = subset_name
        self.batch_size = batch_size

        # Text splitter (same settings)
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". "],
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
        )

        # Chroma setup (same embedding fn)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name,
            device='cuda'
        )
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
        )

    def _load(self):
        return load_dataset(self.dataset_name, self.subset_name)

    def _split_record(self, record: Dict[str, Any]) -> List[Document]:
        text = record["text"]
        source_id = record["id"]
        # chunks = self.splitter.split_text(text)
        chunks = [c for c in self.splitter.split_text(text) if len(c.strip()) >= 20] # only add docs over 20 char long - removes small docs from being added to db

        # preserve original side-effect
        print(f"Document split into {len(chunks)} semantic chunks")

        docs: List[Document] = []
        for chunk in chunks:
            docs.append(
                Document(
                    page_content=chunk.lstrip(". "),
                    metadata={
                        "total_chunks": len(chunks),
                        "chunk_size": len(chunk),
                        "chunk_type": "semantic",
                    },
                    id=source_id, 
                )
            )
        return docs

    def _to_chroma_payload(self, documents):
        ids, texts, metas = [], [], []
        for i, d in enumerate(documents):
            # unique id per chunk
            uid = f"{d.id}-{i}" if getattr(d, "id", None) else str(uuid4())
            # capture the original id and chunk index in metadata
            meta = dict(d.metadata or {})
            meta.update({"source_id": getattr(d, "id", None), "chunk_idx": i})
            ids.append(uid)
            texts.append(d.page_content)
            metas.append(meta)
        return ids, texts, metas

    def _add_documents(self, documents: List[Document]) -> None:
        ids, texts, metas = self._to_chroma_payload(documents)
        for start in range(0, len(ids), self.batch_size):
            sl = slice(start, start + self.batch_size)
            self.collection.add(
                ids=ids[sl],
                documents=texts[sl],
                metadatas=metas[sl],
            )

    def run(self) -> None:
        dataset = self._load()

        # warm up so the model loads/moves to GPU before the big loop
        _ = self.embedding_fn(["__warmup__"])

        for idx, record in enumerate(dataset["train"]):
            docs = self._split_record(record)
            print(f"[stage] adding article {idx} with {len(docs)} chunks", flush=True)
            self._add_documents(docs)


    # def run(self) -> None:
    #     dataset = self._load()
    #     all_docs: List[Document] = []
    #     for record in dataset["train"]:
    #         all_docs.extend(self._split_record(record))
    #     self._add_documents(all_docs)


if __name__ == "__main__":
    WikiChromaIndexer().run()
