import faiss
import numpy as np



# Load existing index
print("Loading existing index...")
index = faiss.read_index('../data/index.faiss')
print(f"Index has {index.ntotal} vectors")

# Try to extract embeddings
try:
    print("Attempting to extract embeddings...")
    
    # For flat indices, we can just reconstruct all
    embeddings = np.zeros((index.ntotal, index.d), dtype=np.float32)
    
    for i in range(index.ntotal):
        if i % 100000 == 0:
            print(f"  Extracted {i}/{index.ntotal} vectors...")
        embeddings[i] = index.reconstruct(i)
    
    print("✅ Successfully extracted embeddings")
    
    # Save them for safekeeping
    np.save('embeddings_extracted.npy', embeddings)
    print("✅ Saved to embeddings_extracted.npy")
    
except Exception as e:
    print(f"❌ Cannot extract: {e}")
    print("   Your index doesn't support reconstruction.")
    print("   You'll need the original embeddings file.")


# Load your embeddings
embeddings = np.load('embeddings.npy')  # Shape: (N, 384)

# Create a FLAT index (supports reconstruction by default)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add vectors
index.add(embeddings)

# Save to a NEW file - don't overwrite the original!
faiss.write_index(index, '../data/flat_index.faiss')
print(f"✅ Created flat index with {index.ntotal} vectors")
print(f"✅ Saved to: index_flat_with_reconstruction.faiss")
print(f"⚠️  Original index.faiss is untouched")