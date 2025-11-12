class Config:
    db_name = 'wiki_data'
    db_path = f"./{db_name}"

    embedding_model = "BAAI/bge-base-en-v1.5"
    reranker = "BAAI/bge-reranker-base"
    generator_model = "Qwen/Qwen2.5-0.5B-Instruct"
