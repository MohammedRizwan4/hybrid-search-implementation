from llama_index.core import SimpleDirectoryReader


def load_documents(folder_path="data"):
    documents = SimpleDirectoryReader(input_dir=folder_path).load_data()
    print(f"Loaded {len(documents)} documents")
    return documents


if __name__ == "__main__":
    load_documents()
