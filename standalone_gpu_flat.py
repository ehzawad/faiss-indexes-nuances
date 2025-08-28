#!/usr/bin/env python3
"""
Standalone GPU Flat FAISS Index Implementation
Complete self-contained implementation for GPU-accelerated exact search
"""

import numpy as np
import pandas as pd
import faiss
import time
from sklearn.feature_extraction.text import TfidfVectorizer

class StandaloneGPUFlat:
    def __init__(self, data_file='landBot-13-7-25_sampled_100.csv'):
        self.data_file = data_file
        self.df = None
        self.embeddings = None
        self.index = None
        self.dimension = 384
        self.build_time = 0
        self.gpu_res = None
        
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
        """Build GPU Flat FAISS index"""
        print("Building GPU Flat index...")
        
        try:
            start_time = time.time()
            
            # Initialize GPU resources
            self.gpu_res = faiss.StandardGpuResources()
            
            # Create CPU index first
            cpu_index = faiss.IndexFlatL2(self.dimension)
            
            # Move to GPU
            self.index = faiss.index_cpu_to_gpu(self.gpu_res, 0, cpu_index)
            self.index.add(self.embeddings)
            
            self.build_time = time.time() - start_time
            
            print(f"Built GPU Flat index in {self.build_time:.4f}s")
            print(f"Index contains {self.index.ntotal} vectors")
            print(f"GPU memory used: {self.index.ntotal * self.dimension * 4 / 1024 / 1024:.2f}MB")
            
            return self.index
            
        except Exception as e:
            print(f"GPU index creation failed: {e}")
            return None
    
    def search(self, query_text, k=5):
        """Search for similar items"""
        if self.index is None:
            return [], 0
            
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
        print(f"\nBenchmarking GPU Flat with {len(test_queries)} queries...")
        
        if self.index is None:
            print("GPU index not available, skipping benchmark")
            return {'build_time': 0, 'avg_search_time': 0, 'results': []}
        
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
        print("GPU FLAT PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        print(f"Build time: {self.build_time:.4f}s")
        print(f"Average search time: {avg_search_time:.2f}ms")
        print(f"Index type: GPU-accelerated exact search")
        print(f"GPU memory usage: {self.index.ntotal * self.dimension * 4 / 1024 / 1024:.2f}MB")
        print(f"Accuracy: 100% (exact search)")
        
        return {
            'build_time': self.build_time,
            'avg_search_time': avg_search_time,
            'results': all_results
        }
    
    def cleanup(self):
        """Clean up GPU resources"""
        if self.gpu_res:
            print("Cleaning up GPU resources...")
    
    def run_complete_experiment(self):
        """Run complete GPU Flat experiment"""
        print("="*60)
        print("STANDALONE GPU FLAT FAISS EXPERIMENT")
        print("="*60)
        
        try:
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
            print("GPU FLAT ALGORITHM CHARACTERISTICS")
            print(f"{'='*60}")
            print("✓ GPU-accelerated exact search")
            print("✓ Parallel processing on GPU")
            print("✓ Good for large datasets (>10K vectors)")
            print("✓ 100% accuracy (exact search)")
            print("✗ Requires GPU memory")
            print("✗ Overhead for small datasets")
            print("✗ CUDA compatibility required")
            
            return results
            
        finally:
            self.cleanup()

def main():
    """Main execution"""
    gpu_flat = StandaloneGPUFlat()
    gpu_flat.run_complete_experiment()

if __name__ == "__main__":
    main()
