#!/usr/bin/env python3
"""
FAISS Index Persistence and Loading System
Demonstrates saving, loading, and managing FAISS indexes
"""

import numpy as np
import pandas as pd
import faiss
import time
import pickle
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class FAISSPersistenceManager:
    def __init__(self, storage_dir: str = "faiss_indexes"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.index_registry = {}
        self.load_registry()
    
    def save_index(self, index: faiss.Index, index_name: str, metadata: Dict = None) -> str:
        """Save FAISS index with metadata"""
        print(f"Saving index: {index_name}")
        
        # Create index-specific directory
        index_dir = self.storage_dir / index_name
        index_dir.mkdir(exist_ok=True)
        
        # Save the FAISS index
        index_path = index_dir / "index.faiss"
        faiss.write_index(index, str(index_path))
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'index_name': index_name,
            'index_type': type(index).__name__,
            'total_vectors': index.ntotal,
            'dimension': index.d,
            'is_trained': index.is_trained,
            'saved_at': time.time(),
            'file_size_bytes': os.path.getsize(index_path)
        })
        
        # Save metadata
        metadata_path = index_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update registry
        self.index_registry[index_name] = {
            'path': str(index_path),
            'metadata_path': str(metadata_path),
            'metadata': metadata
        }
        
        self.save_registry()
        
        print(f"Index saved to: {index_path}")
        print(f"File size: {metadata['file_size_bytes'] / 1024 / 1024:.2f} MB")
        
        return str(index_path)
    
    def load_index(self, index_name: str) -> Tuple[faiss.Index, Dict]:
        """Load FAISS index with metadata"""
        if index_name not in self.index_registry:
            raise ValueError(f"Index '{index_name}' not found in registry")
        
        print(f"Loading index: {index_name}")
        
        registry_entry = self.index_registry[index_name]
        
        # Load the FAISS index
        start_time = time.time()
        index = faiss.read_index(registry_entry['path'])
        load_time = time.time() - start_time
        
        # Load metadata
        with open(registry_entry['metadata_path'], 'r') as f:
            metadata = json.load(f)
        
        metadata['load_time'] = load_time
        
        print(f"Index loaded in {load_time:.4f} seconds")
        print(f"Vectors: {index.ntotal}, Dimension: {index.d}")
        
        return index, metadata
    
    def list_indexes(self) -> Dict:
        """List all saved indexes"""
        print(f"\nSaved FAISS Indexes ({len(self.index_registry)} total):")
        print("-" * 60)
        
        for name, info in self.index_registry.items():
            metadata = info['metadata']
            size_mb = metadata['file_size_bytes'] / 1024 / 1024
            saved_time = time.strftime('%Y-%m-%d %H:%M:%S', 
                                     time.localtime(metadata['saved_at']))
            
            print(f"{name}:")
            print(f"  Type: {metadata['index_type']}")
            print(f"  Vectors: {metadata['total_vectors']}")
            print(f"  Dimension: {metadata['dimension']}")
            print(f"  Size: {size_mb:.2f} MB")
            print(f"  Saved: {saved_time}")
            print()
        
        return self.index_registry
    
    def delete_index(self, index_name: str) -> bool:
        """Delete saved index"""
        if index_name not in self.index_registry:
            print(f"Index '{index_name}' not found")
            return False
        
        registry_entry = self.index_registry[index_name]
        
        # Delete files
        try:
            os.remove(registry_entry['path'])
            os.remove(registry_entry['metadata_path'])
            
            # Remove directory if empty
            index_dir = Path(registry_entry['path']).parent
            if index_dir.exists() and not any(index_dir.iterdir()):
                index_dir.rmdir()
            
            # Remove from registry
            del self.index_registry[index_name]
            self.save_registry()
            
            print(f"Index '{index_name}' deleted successfully")
            return True
            
        except Exception as e:
            print(f"Error deleting index '{index_name}': {e}")
            return False
    
    def save_registry(self):
        """Save index registry to disk"""
        registry_path = self.storage_dir / "registry.json"
        with open(registry_path, 'w') as f:
            json.dump(self.index_registry, f, indent=2)
    
    def load_registry(self):
        """Load index registry from disk"""
        registry_path = self.storage_dir / "registry.json"
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                self.index_registry = json.load(f)
    
    def backup_indexes(self, backup_dir: str = "faiss_backup") -> str:
        """Create backup of all indexes"""
        import shutil
        
        backup_path = Path(backup_dir)
        backup_path.mkdir(exist_ok=True)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        backup_name = f"faiss_backup_{timestamp}"
        full_backup_path = backup_path / backup_name
        
        # Copy entire storage directory
        shutil.copytree(self.storage_dir, full_backup_path)
        
        print(f"Backup created: {full_backup_path}")
        return str(full_backup_path)
    
    def benchmark_persistence(self, index: faiss.Index, index_name: str) -> Dict:
        """Benchmark save/load performance"""
        print(f"Benchmarking persistence for {index_name}...")
        
        # Benchmark save
        start_time = time.time()
        save_path = self.save_index(index, f"{index_name}_benchmark")
        save_time = time.time() - start_time
        
        # Benchmark load
        start_time = time.time()
        loaded_index, metadata = self.load_index(f"{index_name}_benchmark")
        load_time = time.time() - start_time
        
        # Verify integrity
        original_total = index.ntotal
        loaded_total = loaded_index.ntotal
        integrity_ok = original_total == loaded_total
        
        results = {
            'save_time': save_time,
            'load_time': load_time,
            'file_size_mb': metadata['file_size_bytes'] / 1024 / 1024,
            'integrity_check': integrity_ok,
            'compression_ratio': (original_total * index.d * 4) / metadata['file_size_bytes']
        }
        
        print(f"Save time: {save_time:.4f}s")
        print(f"Load time: {load_time:.4f}s")
        print(f"File size: {results['file_size_mb']:.2f} MB")
        print(f"Compression: {results['compression_ratio']:.2f}x")
        print(f"Integrity: {'✓' if integrity_ok else '✗'}")
        
        # Clean up benchmark index
        self.delete_index(f"{index_name}_benchmark")
        
        return results

class FAISSIndexFactory:
    """Factory for creating different types of FAISS indexes"""
    
    @staticmethod
    def create_flat_l2(embeddings: np.ndarray) -> faiss.Index:
        """Create Flat L2 index"""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        return index
    
    @staticmethod
    def create_flat_ip(embeddings: np.ndarray) -> faiss.Index:
        """Create Flat Inner Product index"""
        dimension = embeddings.shape[1]
        embeddings_normalized = embeddings.astype('float32').copy()
        faiss.normalize_L2(embeddings_normalized)
        
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_normalized)
        return index
    
    @staticmethod
    def create_ivf(embeddings: np.ndarray, nlist: int = None) -> faiss.Index:
        """Create IVF index"""
        dimension = embeddings.shape[1]
        if nlist is None:
            nlist = min(100, len(embeddings) // 4)
        
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        
        embeddings_float32 = embeddings.astype('float32')
        index.train(embeddings_float32)
        index.add(embeddings_float32)
        index.nprobe = min(10, nlist)
        
        return index
    
    @staticmethod
    def create_hnsw(embeddings: np.ndarray, M: int = 16) -> faiss.Index:
        """Create HNSW index"""
        dimension = embeddings.shape[1]
        index = faiss.IndexHNSWFlat(dimension, M)
        index.hnsw.efConstruction = 40
        index.add(embeddings.astype('float32'))
        index.hnsw.efSearch = 16
        return index

def demonstrate_persistence():
    """Demonstrate FAISS persistence capabilities"""
    print("FAISS Persistence Demonstration")
    print("=" * 50)
    
    # Load sample data
    df = pd.read_csv('landBot-13-7-25_sampled_100.csv')
    
    # Create sample embeddings
    dimension = 384
    num_samples = len(df)
    embeddings = np.random.randn(num_samples, dimension).astype('float32')
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    # Initialize persistence manager
    manager = FAISSPersistenceManager()
    
    # Create and save different index types
    print("\n1. Creating and saving indexes...")
    
    # Flat L2 Index
    flat_l2 = FAISSIndexFactory.create_flat_l2(embeddings)
    manager.save_index(flat_l2, "demo_flat_l2", {
        'description': 'Exact L2 distance search',
        'dataset': 'Bengali Q&A sample',
        'created_by': 'FAISS Research Demo'
    })
    
    # HNSW Index
    hnsw = FAISSIndexFactory.create_hnsw(embeddings)
    manager.save_index(hnsw, "demo_hnsw", {
        'description': 'Approximate graph-based search',
        'M': 16,
        'efSearch': 16
    })
    
    # IVF Index
    ivf = FAISSIndexFactory.create_ivf(embeddings)
    manager.save_index(ivf, "demo_ivf", {
        'description': 'Inverted file approximate search',
        'nlist': ivf.nlist,
        'nprobe': ivf.nprobe
    })
    
    # List all saved indexes
    print("\n2. Listing saved indexes...")
    manager.list_indexes()
    
    # Load and test an index
    print("\n3. Loading and testing index...")
    loaded_index, metadata = manager.load_index("demo_hnsw")
    
    # Test search
    query = embeddings[:1]
    distances, indices = loaded_index.search(query, 5)
    print(f"Search test - Found {len(indices[0])} results")
    
    # Benchmark persistence performance
    print("\n4. Benchmarking persistence performance...")
    for index_name, index in [("FlatL2", flat_l2), ("HNSW", hnsw), ("IVF", ivf)]:
        results = manager.benchmark_persistence(index, index_name)
        print()
    
    # Create backup
    print("\n5. Creating backup...")
    backup_path = manager.backup_indexes()
    
    # Cleanup demonstration indexes
    print("\n6. Cleaning up...")
    for index_name in ["demo_flat_l2", "demo_hnsw", "demo_ivf"]:
        manager.delete_index(index_name)
    
    print("\nPersistence demonstration completed!")

def main():
    """Main execution function"""
    demonstrate_persistence()

if __name__ == "__main__":
    main()
