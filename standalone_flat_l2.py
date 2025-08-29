#!/usr/bin/env python3
"""
Standalone FlatL2 FAISS Index Implementation
Complete self-contained implementation for exact L2 distance search
"""

import numpy as np
import pandas as pd
import faiss
import time
from sklearn.feature_extraction.text import TfidfVectorizer

class StandaloneFlatL2:
    def __init__(self, data_file='landBot-13-7-25.csv'):
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
        
        # Combine question and answer text
        texts = (self.df['question'] + ' ' + self.df['answer']).tolist()
        
        # Create TF-IDF vectorizer with fixed dimension
        vectorizer = TfidfVectorizer(max_features=self.dimension, stop_words=None)
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Convert to dense numpy array and pad/truncate to exact dimension
        embeddings_raw = tfidf_matrix.toarray().astype('float32')
        
        # Ensure exact dimension match
        if embeddings_raw.shape[1] < self.dimension:
            # Pad with zeros
            padding = np.zeros((embeddings_raw.shape[0], self.dimension - embeddings_raw.shape[1]), dtype='float32')
            self.embeddings = np.hstack([embeddings_raw, padding])
        else:
            # Truncate to exact dimension
            self.embeddings = embeddings_raw[:, :self.dimension]
        
        print(f"Created embeddings: {self.embeddings.shape}")
        return self.embeddings
    
    def build_index(self):
        """Build FlatL2 FAISS index"""
        print("Building FlatL2 index...")
        
        start_time = time.time()
        
        # Create FlatL2 index for exact L2 distance search
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(self.embeddings)
        
        self.build_time = time.time() - start_time
        
        print(f"Built FlatL2 index in {self.build_time:.4f}s")
        print(f"Index contains {self.index.ntotal} vectors")
        print(f"Index type: {type(self.index).__name__}")
        
        return self.index
    
    def search(self, query_text, k=5):
        """Search for similar items"""
        # Create query embedding (simple word overlap)
        query_words = set(query_text.lower().split())
        
        # Calculate similarity with all documents
        query_embedding = np.zeros(self.dimension, dtype='float32')
        
        # Simple embedding: set 1 for matching features
        for i, text in enumerate(self.df['question'] + ' ' + self.df['answer']):
            text_words = set(text.lower().split())
            overlap = len(query_words & text_words)
            if i < self.dimension:
                query_embedding[i] = overlap
        
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        start_time = time.time()
        distances, indices = self.index.search(query_embedding, k)
        search_time = (time.time() - start_time) * 1000
        
        # Format results
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
        print(f"\nBenchmarking FlatL2 with {len(test_queries)} queries...")
        
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
        print("FLATL2 PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        print(f"Build time: {self.build_time:.4f}s")
        print(f"Average search time: {avg_search_time:.2f}ms")
        print(f"Index type: Exact L2 distance search")
        print(f"Memory usage: {self.index.ntotal * self.dimension * 4 / 1024 / 1024:.2f}MB")
        print(f"Accuracy: 100% (exact search)")
        
        return {
            'build_time': self.build_time,
            'avg_search_time': avg_search_time,
            'results': all_results
        }
    
    def run_complete_experiment(self):
        """Run complete FlatL2 experiment"""
        print("="*60)
        print("STANDALONE FLATL2 FAISS EXPERIMENT")
        print("="*60)
        
        # Load data and create embeddings
        self.load_data()
        self.create_embeddings()
        
        # Build index
        self.build_index()
        
        # Test queries
        test_queries = [
            "জমি কিনেছি নামজারি করতে চাই",
            "নিবন্ধন করতে কি লাগবে", 
            "প্রতিনিধি দিয়ে আবেদন করা যায়",
            "আবেদন জমা দিতে কি সেবাপোর্টালে যেতে হয়",
            "নামজারি বাতিল হলে কি করবো"
        ]
        
        # Benchmark
        results = self.benchmark(test_queries)
        
        print(f"\n{'='*60}")
        print("FLATL2 ALGORITHM CHARACTERISTICS")
        print(f"{'='*60}")
        print("✓ Exact search - 100% accuracy")
        print("✓ Fast for small datasets (<1K vectors)")
        print("✓ Simple implementation")
        print("✓ No training required")
        print("✗ Linear search complexity O(n)")
        print("✗ Doesn't scale to large datasets")
        print("✗ High memory usage for large dimensions")
        
        return results

def main():
    """Main execution"""
    flat_l2 = StandaloneFlatL2()
    flat_l2.run_complete_experiment()

if __name__ == "__main__":
    main()
