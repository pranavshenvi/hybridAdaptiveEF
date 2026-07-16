import os
import urllib.request
import zipfile
import numpy as np

GLOVE_ZIP_URL = "http://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_ZIP_PATH = "glove.6B.zip"
GLOVE_TXT_PATH = "glove.6B.100d.txt"

def main():
    print("=" * 80)
    print("  GloVe Data Preparation (Authors' Procedure)")
    print("=" * 80)
    
    # 1. Download zip if needed
    if not os.path.exists(GLOVE_TXT_PATH):
        if not os.path.exists(GLOVE_ZIP_PATH):
            print(f"Downloading {GLOVE_ZIP_URL} (this is ~822MB, please wait)...")
            urllib.request.urlretrieve(GLOVE_ZIP_URL, GLOVE_ZIP_PATH)
            print("Download complete.")
        
        print(f"Extracting {GLOVE_TXT_PATH} from zip...")
        with zipfile.ZipFile(GLOVE_ZIP_PATH, 'r') as zip_ref:
            # Only extract the 100d file to save space
            zip_ref.extract(GLOVE_TXT_PATH, path=".")
        print("Extraction complete.")
    else:
        print(f"{GLOVE_TXT_PATH} already exists.")

    # 2. Read text vectors
    print("Reading text vectors from raw GloVe file...")
    raw_dataset_vectors = []
    with open(GLOVE_TXT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 101: 
                continue
            # The first part is the word, the rest are float coordinates
            vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            raw_dataset_vectors.append(vector)
            
    print(f"Loaded {len(raw_dataset_vectors)} raw vectors.")

    # 3. Shuffle exactly like authors (but seeded for reproducibility)
    print("Shuffling vectors and extracting splits...")
    np.random.seed(42)  # Added seed so it's a fair, reproducible test
    np.random.shuffle(raw_dataset_vectors)
    
    training_vectors_num = 350000
    query_vectors_num = 10000
    
    train_queries = np.array(raw_dataset_vectors[:training_vectors_num], dtype=np.float32)
    test_queries = np.array(raw_dataset_vectors[training_vectors_num : training_vectors_num + query_vectors_num], dtype=np.float32)
    
    # 4. Save to disk as fast-loading NPZ files
    print(f"Saving {len(train_queries)} training queries to glove_learn.npz")
    np.savez("glove_learn.npz", emb=train_queries)
    
    print(f"Saving {len(test_queries)} testing queries to glove_query.npz")
    np.savez("glove_query.npz", emb=test_queries)
    
    print("Preparation completely successful!")

if __name__ == "__main__":
    main()
