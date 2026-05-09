import argparse

from source_retrieval_branches import embeddings_lookup


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-id", required=True)
    args = parser.parse_args()

    result_df = embeddings_lookup(args.doc_id)

    print("\nTOP EMBEDDING RESULTS")
    print(result_df.to_string(index=False))