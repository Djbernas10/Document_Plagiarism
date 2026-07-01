import pandas as pd
from pathlib import Path

shards_path = Path("artifacts/embeddings/embeddings_qwen06b/faiss_shards.parquet")
df = pd.read_parquet(shards_path)

df["shard_path"] = df["shard_path"].apply(
    lambda p: "/workspace/Document_Plagiarism/artifacts/embeddings/embeddings_qwen06b/shards/" + Path(p).name
)

df.to_parquet(shards_path, index=False)
print(df["shard_path"].tolist())