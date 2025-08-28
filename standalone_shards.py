#!/usr/bin/env python3
"""
Standalone Shards FAISS Index Implementation
Complete self-contained implementation for distributed multi-index search
"""

import numpy as np
import pandas as pd
import faiss
import time
from sklearn.feature_extraction.text import TfidfVectorizer

class StandaloneShards:
    def __init__(self, data_file='landBot-13-7-25_sampled_100.csv'):
        self.data_file = data_file
        self.df = None
        self.embeddings = None
        self.index = None
        self.dimension = 384
        self.build_time = 0
        self.num_shards = 4
        
    def load_data(self):
        """Load Bengali Q&A dataset"""
        print("Loading Bengali Q&A dataset...")
        self.df = pd.read_csv(self.data_file)
        print(f"Loaded {len(self.df)} samples with {len(self.df['tag'].unique())} unique tags")
        return self.df
    
    def create_embeddings(self):
        """Create TF-IDF style embeddings"""
        print("Creating TF-IDF embeddings...")
        
        texts = (self.df['question'] + ' ' + self.df['answer']).tolist()
        vectorizer = TfidfVectorizer(max_features=self.dimension, stop_words=None)
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        embeddings_raw = tfidf_matrix.toarray().astype('float32')
        
        if embeddings_raw.shape[1] < self.dimension:
            padding = np.zeros((embeddings_raw.shape[0], self.dimension - embeddings_raw.shape[1]), dtype='float32')
            self.embeddings = np.hstack([embeddings_raw, padding])
        else:
            self.embeddings = embeddings_raw[:, :self.dimension]
        
        print(f"Created embeddings: {self.embeddings.shape}")
        return self.embeddings
    
    def build_index(self):
        """Build Shards FAISS index"""
        print(f"Building Shards index with {self.num_shards} shards...")
        
        start_time = time.time()
        
        # Create sharded index
        self.index = faiss.IndexShards(self.dimension)
        
        # Create individual shard indexes
        shard_size = len(self.embeddings) // self.num_shards
        
        for i in range(self.num_shards):
            start_idx = i * shard_size
            if i == self.num_shards - 1:
                end_idx = len(self.embeddings)
            else:
                end_idx = (i + 1) * shard_size
            
            # Create shard index
            shard_index = faiss.IndexFlatL2(self.dimension)
            shard_data = self.embeddings[start_idx:end_idx]
            shard_index.add(shard_data)
            
            # Add to shards
            self.index.add_shard(shard_index)
            print(f"  Shard {i+1}: {len(shard_data)} vectors")
        
        self.build_time = time.time() - start_time
        
        print(f"Built Shards index in {self.build_time:.4f}s")
        print(f"Index contains {self.index.ntotal} vectors across {self.num_shards} shards")
        
        return self.index
    
    def search(self, query_text, k=5):
        """Search across all shards"""
        query_words = set(query_text.lower().split())
        query_embedding = np.zeros(self.dimension, dtype='float32')
        
        for i, text in enumerate(self.df['question'] + ' ' + self.df['answer']):
            text_words = set(text.lower().split())
            overlap = len(query_words & text_words)
            if i < self.dimension:
                query_embedding[i] = overlap
        
        query_embedding = query_embedding.reshape(1, -1)
        
        start_time = time.time()
        distances, indices = self.index.search(query_embedding, k)
        search_time = (time.time() - start_time) * 1000
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            results.append({
                'rank': i + 1,
                'distance': float(dist),
                'question': self.df.iloc[idx]['question'],
                'answer': self.df.iloc[idx]['answer'],
                'tag': self.df.iloc[idx]['tag']
            })
        
        return results, search_time
    
    def benchmark(self, test_queries):
        """Benchmark the index with test queries"""
        print(f"\nBenchmarking Shards with {len(test_queries)} queries...")
        
        total_search_time = 0
        all_results = []
        
        for i, query in enumerate(test_queries):
            print(f"\nQuery {i+1}: {query}")
            results, search_time = self.search(query)
            total_search_time += search_time
            all_results.append(results)
            
            print(f"Search time: {search_time:.2f}ms")
            print("Top 3 results:")
            for result in results[:3]:
                print(f"  {result['rank']}. [{result['distance']:.3f}] {result['question'][:80]}...")
        
        avg_search_time = total_search_time / len(test_queries)
        
        print(f"\n{'='*60}")
        print("SHARDS PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        print(f"Build time: {self.build_time:.4f}s")
        print(f"Average search time: {avg_search_time:.2f}ms")
        print(f"Index type: Distributed multi-shard search")
        print(f"Number of shards: {self.num_shards}")
        print(f"Memory usage: {self.index.ntotal * self.dimension * 4 / 1024 / 1024:.2f}MB")
        
        return {
            'build_time': self.build_time,
            'avg_search_time': avg_search_time,
            'results': all_results
        }
    
    def run_complete_experiment(self):
        """Run complete Shards experiment"""
        print("="*60)
        print("STANDALONE SHARDS FAISS EXPERIMENT")
        print("="*60)
        
        self.load_data()
        self.create_embeddings()
        self.build_index()
        
        test_queries = [
            "জমি কিনেছি নামজারি করতে চাই",
            "নিবন্ধন করতে কি লাগবে", 
            "প্রতিনিধি দিয়ে আবেদন করা যায়",
            "আবেদন জমা দিতে কি সেবাপোর্টালে যেতে হয়",
            "নামজারি বাতিল হলে কি করবো"
        ]
        
        results = self.benchmark(test_queries)
        
        print(f"\n{'='*60}")
        print("SHARDS ALGORITHM CHARACTERISTICS")
        print(f"{'='*60}")
        print("✓ Distributed search across multiple indexes")
        print("✓ Good for horizontal scaling")
        print("✓ Parallelizable across machines")
        print("✓ Fault tolerance (shard isolation)")
        print("✗ Complexity in shard management")
        print("✗ Load balancing challenges")
        print("✗ Network overhead in distributed setup")
        
        return results

def main():
    """Main execution"""
    shards = StandaloneShards()
    shards.run_complete_experiment()

if __name__ == "__main__":
    main()
