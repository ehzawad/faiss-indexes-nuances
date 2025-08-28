#!/usr/bin/env python3
"""
Standalone IDMap FAISS Index Implementation
Complete self-contained implementation for ID-preserving wrapper index
"""

import numpy as np
import pandas as pd
import faiss
import time
from sklearn.feature_extraction.text import TfidfVectorizer

class StandaloneIDMap:
    def __init__(self, data_file='landBot-13-7-25_sampled_100.csv'):
        self.data_file = data_file
        self.df = None
        self.embeddings = None
        self.index = None
        self.dimension = 384
        self.build_time = 0
        self.custom_ids = None
        
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
        
        # Create custom IDs (e.g., hash of question + tag)
        self.custom_ids = []
        for i, row in self.df.iterrows():
            custom_id = hash(row['question'] + row['tag']) % 1000000
            self.custom_ids.append(custom_id)
        
        self.custom_ids = np.array(self.custom_ids, dtype=np.int64)
        
        print(f"Created embeddings: {self.embeddings.shape}")
        print(f"Created custom IDs: {len(self.custom_ids)} unique IDs")
        return self.embeddings
    
    def build_index(self):
        """Build IDMap FAISS index"""
        print("Building IDMap index...")
        
        start_time = time.time()
        
        # Create base index
        base_index = faiss.IndexFlatL2(self.dimension)
        
        # Wrap with IDMap
        self.index = faiss.IndexIDMap(base_index)
        
        # Add with custom IDs
        self.index.add_with_ids(self.embeddings, self.custom_ids)
        
        self.build_time = time.time() - start_time
        
        print(f"Built IDMap index in {self.build_time:.4f}s")
        print(f"Index contains {self.index.ntotal} vectors")
        print(f"Custom ID mapping enabled")
        
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
        
        start_time = time.time()
        distances, indices = self.index.search(query_embedding, k)
        search_time = (time.time() - start_time) * 1000
        
        results = []
        for i, (dist, custom_id) in enumerate(zip(distances[0], indices[0])):
            # Find original row by custom ID
            original_idx = None
            for j, cid in enumerate(self.custom_ids):
                if cid == custom_id:
                    original_idx = j
                    break
            
            if original_idx is not None:
                results.append({
                    'rank': i + 1,
                    'distance': float(dist),
                    'custom_id': int(custom_id),
                    'question': self.df.iloc[original_idx]['question'],
                    'answer': self.df.iloc[original_idx]['answer'],
                    'tag': self.df.iloc[original_idx]['tag']
                })
        
        return results, search_time
    
    def benchmark(self, test_queries):
        """Benchmark the index with test queries"""
        print(f"\nBenchmarking IDMap with {len(test_queries)} queries...")
        
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
                print(f"  {result['rank']}. [ID: {result['custom_id']}] [{result['distance']:.3f}] {result['question'][:70]}...")
        
        avg_search_time = total_search_time / len(test_queries)
        
        print(f"\n{'='*60}")
        print("IDMAP PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        print(f"Build time: {self.build_time:.4f}s")
        print(f"Average search time: {avg_search_time:.2f}ms")
        print(f"Index type: ID-preserving wrapper")
        print(f"Memory usage: {self.index.ntotal * self.dimension * 4 / 1024 / 1024:.2f}MB")
        print(f"Custom ID mapping: Enabled")
        
        return {
            'build_time': self.build_time,
            'avg_search_time': avg_search_time,
            'results': all_results
        }
    
    def run_complete_experiment(self):
        """Run complete IDMap experiment"""
        print("="*60)
        print("STANDALONE IDMAP FAISS EXPERIMENT")
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
        print("IDMAP ALGORITHM CHARACTERISTICS")
        print(f"{'='*60}")
        print("✓ Preserves custom ID mapping")
        print("✓ Wraps any base index type")
        print("✓ Essential for production systems")
        print("✓ Maintains external ID references")
        print("✗ Slight memory overhead for ID storage")
        print("✗ ID lookup adds minor latency")
        print("✗ Requires careful ID management")
        
        return results

def main():
    """Main execution"""
    idmap = StandaloneIDMap()
    idmap.run_complete_experiment()

if __name__ == "__main__":
    main()
