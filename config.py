class Config:
    dataset_name = "wikimedia/wikipedia"
    subset_name = "20231101.simple"

    db_name = 'wiki-data'
    db_path = f"./{db_name}"

    collection_name = 'wiki'

    embedding_model = "BAAI/bge-base-en-v1.5"
    reranker = "BAAI/bge-reranker-base"
    generator_model = "Qwen/Qwen2.5-0.5B-Instruct"
