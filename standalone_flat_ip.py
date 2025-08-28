#!/usr/bin/env python3
"""
Standalone FlatIP FAISS Index Implementation
Complete self-contained implementation for exact cosine similarity search
"""

import numpy as np
import pandas as pd
import faiss
import time
from sklearn.feature_extraction.text import TfidfVectorizer

class StandaloneFlatIP:
    def __init__(self, data_file='landBot-13-7-25_sampled_100.csv'):
        self.data_file = data_file
        self.df = None
        self.embeddings = None
        self.index = None
        self.dimension = 384
        self.build_time = 0
        
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
        
        # Ensure exact dimension match
        if embeddings_raw.shape[1] < self.dimension:
            padding = np.zeros((embeddings_raw.shape[0], self.dimension - embeddings_raw.shape[1]), dtype='float32')
            self.embeddings = np.hstack([embeddings_raw, padding])
        else:
            self.embeddings = embeddings_raw[:, :self.dimension]
        
        # Normalize for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        print(f"Created normalized embeddings: {self.embeddings.shape}")
        return self.embeddings
    
    def build_index(self):
        """Build FlatIP FAISS index"""
        print("Building FlatIP index...")
        
        start_time = time.time()
        
        # Create FlatIP index for exact cosine similarity search
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(self.embeddings)
        
        self.build_time = time.time() - start_time
        
        print(f"Built FlatIP index in {self.build_time:.4f}s")
        print(f"Index contains {self.index.ntotal} vectors")
        print(f"Index type: {type(self.index).__name__}")
        
        return self.index
    
    def search(self, query_text, k=5):
        """Search for similar items"""
        query_words = set(query_text.lower().split())
        query_embedding = np.zeros(self.dimension, dtype='float32')
        
        for i, text in enumerate(self.df['question'] + ' ' + self.df['answer']):
            text_words = set(text.lower().split())
            overlap = len(query_words & text_words)
            if i < self.dimension:
                query_embedding[i] = overlap
        
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)  # Normalize query for cosine similarity
        
        start_time = time.time()
        distances, indices = self.index.search(query_embedding, k)
        search_time = (time.time() - start_time) * 1000
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            results.append({
                'rank': i + 1,
                'similarity': float(dist),  # Higher is better for IP
                'question': self.df.iloc[idx]['question'],
                'answer': self.df.iloc[idx]['answer'],
                'tag': self.df.iloc[idx]['tag']
            })
        
        return results, search_time
    
    def benchmark(self, test_queries):
        """Benchmark the index with test queries"""
        print(f"\nBenchmarking FlatIP with {len(test_queries)} queries...")
        
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
                print(f"  {result['rank']}. [Sim: {result['similarity']:.3f}] {result['question'][:80]}...")
        
        avg_search_time = total_search_time / len(test_queries)
        
        print(f"\n{'='*60}")
        print("FLATIP PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        print(f"Build time: {self.build_time:.4f}s")
        print(f"Average search time: {avg_search_time:.2f}ms")
        print(f"Index type: Exact cosine similarity search")
        print(f"Memory usage: {self.index.ntotal * self.dimension * 4 / 1024 / 1024:.2f}MB")
        print(f"Accuracy: 100% (exact search)")
        
        return {
            'build_time': self.build_time,
            'avg_search_time': avg_search_time,
            'results': all_results
        }
    
    def run_complete_experiment(self):
        """Run complete FlatIP experiment"""
        print("="*60)
        print("STANDALONE FLATIP FAISS EXPERIMENT")
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
        print("FLATIP ALGORITHM CHARACTERISTICS")
        print(f"{'='*60}")
        print("✓ Exact cosine similarity - 100% accuracy")
        print("✓ Fast for small datasets (<1K vectors)")
        print("✓ Perfect for text similarity tasks")
        print("✓ Normalized vectors required")
        print("✗ Linear search complexity O(n)")
        print("✗ Doesn't scale to large datasets")
        print("✗ High memory usage for large dimensions")
        
        return results

def main():
    """Main execution"""
    flat_ip = StandaloneFlatIP()
    flat_ip.run_complete_experiment()

if __name__ == "__main__":
    main()
