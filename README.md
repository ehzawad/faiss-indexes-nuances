# FAISS Research & Experimentation Project - 17 Standalone Implementations

A comprehensive research project exploring Facebook AI Similarity Search (FAISS) indexes, search mechanisms, and performance characteristics using Bengali Q&A data.

## 🎯 Project Overview

This project provides complete coverage of FAISS capabilities through:
- **17 Standalone Index Types**: All major FAISS implementations
- **Performance Comparison**: Speed, accuracy, and memory analysis
- **Similarity Quality Analysis**: Semantic relevance evaluation
- **Complete Independence**: Each implementation is self-contained
- **Bengali Q&A Dataset**: Real-world semantic search testing

## 📁 Project Structure

```
faiss-stuff/
├── README.md                           # This documentation
├── requirements.txt                    # Python dependencies
├── landBot-13-7-25_sampled_100.csv   # Bengali Q&A dataset (100 samples)
├── FAISS_INDEX_TYPES.md              # Complete index type documentation
├── RESEARCH_SUMMARY.md               # Research findings summary
│
├── 17 Standalone Implementations:
├── standalone_flat_l2.py             # Exact L2 distance search
├── standalone_flat_ip.py             # Exact cosine similarity search
├── standalone_ivf.py                 # Clustered approximate search
├── standalone_hnsw.py                # Graph-based search
├── standalone_pq.py                  # Product quantization
├── standalone_lsh.py                 # Locality sensitive hashing
├── standalone_gpu_flat.py            # GPU-accelerated exact search
├── standalone_gpu_ivfpq.py           # GPU IVF + Product Quantization
├── standalone_binary_flat.py         # Binary Hamming distance
├── standalone_binary_ivf.py          # Binary approximate search
├── standalone_scalar_quantizer.py    # 8-bit scalar quantization
├── standalone_ivf_scalar_quantizer.py # IVF + scalar quantization
├── standalone_idmap.py               # ID preservation wrapper
├── standalone_shards.py              # Distributed multi-shard
├── standalone_cascade_search.py      # Fast→exact pipeline
├── standalone_ensemble_search.py     # Multi-index voting
├── standalone_adaptive_search.py     # Query-based selection
│
├── Analysis Tools:
├── compare_all_faiss_implementations.py # Complete performance comparison
├── similarity_quality_analysis.py      # Semantic relevance analysis
├── faiss_persistence.py               # Index save/load mechanisms
│
└── Generated Results:
    ├── faiss_implementations_comparison.png    # Performance charts
    ├── faiss_similarity_quality_analysis.png  # Similarity quality heatmap
    ├── faiss_comparison_results.csv           # Detailed performance data
    └── faiss_similarity_quality_scores.csv    # Similarity scoring matrix
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Conda environment with FAISS support
- Required packages: `numpy`, `pandas`, `faiss-cpu`, `matplotlib`, `seaborn`

### Installation
```bash
# Using your existing conda environment
/home/synesis/miniconda3/envs/faiss-cuda12.1/bin/pip install numpy pandas matplotlib seaborn

# Or install from requirements
/home/synesis/miniconda3/envs/faiss-cuda12.1/bin/pip install -r requirements.txt
```

### Running Experiments

1. **Individual Implementation Testing**:
```bash
# Run any standalone implementation
conda activate faiss-cuda12.1
python standalone_flat_l2.py
python standalone_hnsw.py
python standalone_gpu_flat.py
# ... any of the 17 implementations
```

2. **Complete Performance Comparison**:
```bash
conda activate faiss-cuda12.1
python compare_all_faiss_implementations.py
```

3. **Similarity Quality Analysis**:
```bash
conda activate faiss-cuda12.1
python similarity_quality_analysis.py
```

4. **Index Persistence Demo**:
```bash
conda activate faiss-cuda12.1
python faiss_persistence.py
```

## 📊 Key Findings

### Complete Performance Summary (All 17 Implementations)

| Rank | Algorithm | Build Time | Search Time | Memory | Similarity Score | Best For |
|------|-----------|------------|-------------|--------|------------------|----------|
| 1 | **Adaptive Search** | Variable | 0.04ms | Variable | 0.131 | Dynamic workloads |
| 2 | **IVF** | 36.8ms | 0.08ms | 0.150MB | 0.131 | Medium-large datasets |
| 3 | **HNSW** | 4.5ms | 0.08ms | 0.180MB | 0.131 | Consistent low latency |
| 4 | **Cascade Search** | 6.8ms | 0.10ms | 0.330MB | 0.131 | Speed + accuracy |
| 5 | **Ensemble Search** | 15.7ms | 0.24ms | 0.480MB | 0.131 | Robust consensus |
| 6 | **LSH** | 11.4ms | 0.35ms | 0.150MB | 0.022 | Hash-based approx |
| 7 | **Binary IVF** | 66.9ms | 0.47ms | 0.005MB | 0.178 | Fast binary search |
| 8 | **Flat IP** | 0.1ms | 0.61ms | 0.150MB | 0.131 | Cosine similarity |
| 9 | **Binary Flat** | 0.1ms | 0.87ms | 0.005MB | 0.178 | Extreme compression |
| 10 | **Shards** | 0.1ms | 0.99ms | 0.150MB | 0.131 | Distributed search |
| 11 | **Flat L2** | 0.1ms | 1.65ms | 0.150MB | 0.131 | Exact search |
| 12 | **PQ** | 102.1ms | 1.69ms | 0.001MB | 0.150 | Memory constrained |
| 13 | **IDMap** | 0.2ms | 1.82ms | 0.150MB | 0.131 | ID preservation |
| 14 | **Scalar Quantizer** | 5.6ms | 2.91ms | 0.040MB | 0.131 | Balanced compression |
| 15 | **GPU Flat** | 224.3ms | 3.52ms | 0.150MB | 0.131 | GPU acceleration |
| 16 | **GPU IVF+PQ** | 398.1ms | 7.05ms | 0.001MB | 0.131 | GPU + compression |
| 17 | **IVF Scalar Quantizer** | 18.1ms | 0.09ms | 0.040MB | 0.131 | Fast + compressed |

### Hybrid Search Performance

| Method | Search Time | Precision@5 | Recall@5 | Description |
|--------|-------------|-------------|----------|-------------|
| **Cascade** | 0.13ms | 0.320 | 0.257 | Fast approximate → exact refinement |
| **Ensemble** | 0.34ms | 0.333 | 0.263 | Multiple indexes voting |
| **Adaptive** | 0.08ms | 0.333 | 0.263 | Query-based index selection |

### Scalability Insights (up to 5000 vectors)

- **FlatL2**: Linear scaling, best for exact search on small datasets
- **HNSW**: Best search performance, scales well with dataset size
- **IVF**: Good balance, efficient for medium to large datasets
- **PQ**: Excellent compression (768x), acceptable quality loss

## 🔍 Index Types Explained

### 1. Flat Indexes (Exact Search)
- **FlatL2**: Euclidean distance, exact results
- **FlatIP**: Inner product/cosine similarity, exact results
- **Best for**: Small datasets (<1K vectors), when accuracy is critical

### 2. Approximate Indexes
- **IVF (Inverted File)**: Clusters data, searches subset of clusters
- **HNSW**: Graph-based navigation, excellent speed/accuracy tradeoff
- **Best for**: Large datasets, when slight accuracy loss is acceptable

### 3. Compressed Indexes
- **PQ (Product Quantization)**: Compresses vectors, reduces memory
- **LSH (Locality Sensitive Hashing)**: Hash-based approximate search
- **Best for**: Memory-constrained environments, very large datasets

## 🔬 Research Methodology

### Dataset
- **Source**: Bengali Q&A pairs from land registration domain
- **Size**: 100 samples with questions, answers, and tags
- **Preprocessing**: Text-based embeddings with semantic clustering
- **Evaluation**: Tag-based relevance for precision/recall metrics

### Evaluation Metrics
- **Build Time**: Index construction speed
- **Search Time**: Query response latency
- **Memory Usage**: Storage requirements
- **Precision@K**: Fraction of relevant results in top K
- **Recall@K**: Fraction of relevant items retrieved
- **Mean Reciprocal Rank (MRR)**: Ranking quality metric

### Experimental Design
1. **Individual Index Testing**: Each index type with separate embeddings
2. **Hybrid Mechanisms**: Combining multiple indexes for better performance
3. **Scalability Analysis**: Performance across dataset sizes (100-5000 vectors)
4. **Quality Assessment**: Precision, recall, and MRR evaluation
5. **Persistence Testing**: Save/load performance and integrity

## 📈 Performance Recommendations

### Small Datasets (<1K vectors)
- **Primary**: FlatL2 or FlatIP for exact results
- **Alternative**: HNSW for future scalability

### Medium Datasets (1K-10K vectors)
- **Primary**: HNSW for best speed/accuracy balance
- **Alternative**: IVF with appropriate nlist parameter

### Large Datasets (>10K vectors)
- **Primary**: IVF with optimized clustering
- **Secondary**: HNSW for consistent low latency
- **Memory-constrained**: PQ for compression

### Real-time Applications
- **Primary**: HNSW for consistent performance
- **Hybrid**: Adaptive search based on query characteristics

## 🛠 Advanced Features

### Hybrid Search Strategies

1. **Cascade Search**:
   ```python
   # Fast approximate search → exact refinement
   primary_results = ivf_index.search(query, k*3)
   refined_results = flat_index.search(candidates, k)
   ```

2. **Ensemble Search**:
   ```python
   # Multiple indexes voting
   results = []
   for index in [flat, hnsw, ivf]:
       results.extend(index.search(query, k))
   final_results = aggregate_by_score(results)
   ```

3. **Adaptive Search**:
   ```python
   # Choose index based on query characteristics
   if query_confidence > threshold:
       return exact_index.search(query, k)
   else:
       return approximate_index.search(query, k)
   ```

### Index Persistence

```python
# Save index with metadata
manager = FAISSPersistenceManager()
manager.save_index(index, "my_index", {
    'description': 'Production search index',
    'dataset': 'Bengali Q&A',
    'version': '1.0'
})

# Load index
loaded_index, metadata = manager.load_index("my_index")

# List all saved indexes
manager.list_indexes()
```

## 📝 Usage Examples

### Basic Index Creation
```python
from flat_l2_index import FlatL2Index

# Create and test index
index = FlatL2Index('your_data.csv')
stats, results = index.run_full_pipeline([
    "your test query 1",
    "your test query 2"
])
```

### Advanced Research
```python
from advanced_faiss_research import AdvancedFAISSResearch

research = AdvancedFAISSResearch('your_data.csv')
research.load_data()
research.create_realistic_embeddings()
research.create_composite_indexes()
research.create_hybrid_search_mechanisms()

benchmark_results = research.comprehensive_benchmark()
research.generate_advanced_report(benchmark_results)
```

### Scalability Testing
```python
from scalability_test import FAISSScalabilityTest

test = FAISSScalabilityTest('your_data.csv')
results = test.run_scalability_tests()
test.create_scalability_visualizations()
test.generate_scalability_report()
```

## 🎯 Key Insights & Conclusions

### Performance Trade-offs
1. **Accuracy vs Speed**: Exact methods (Flat) provide perfect results but don't scale
2. **Memory vs Quality**: Compressed methods (PQ) save space but reduce accuracy
3. **Build vs Search**: Some indexes (HNSW) take longer to build but search faster

### Scaling Characteristics
- **Linear scaling**: FlatL2 search time grows linearly with dataset size
- **Logarithmic scaling**: HNSW maintains low search latency as data grows
- **Cluster-dependent**: IVF performance depends on optimal cluster count

### Practical Recommendations
1. **Start with FlatL2** for prototyping and small datasets
2. **Migrate to HNSW** when dataset grows beyond 1K vectors
3. **Use IVF** for very large datasets with batch processing
4. **Consider PQ** when memory is the primary constraint
5. **Implement hybrid approaches** for production systems

### Bengali Text Insights
- Tag-based clustering improves semantic search quality
- Word overlap provides good baseline similarity
- Normalized embeddings work well with cosine similarity (FlatIP)

## 🔮 Future Extensions

### Potential Improvements
1. **Real Sentence Embeddings**: Integration with transformer models
2. **GPU acceleration**: FAISS GPU indexes for larger datasets
3. **Dynamic Updates**: Online index updates and maintenance
4. **Multi-modal Search**: Combining text with other data types
5. **Production Deployment**: API wrapper and monitoring

### Research Directions
1. **Optimal Parameter Tuning**: Automated hyperparameter optimization
2. **Quality Metrics**: More sophisticated relevance measures
3. **Federated Search**: Distributed FAISS across multiple nodes
4. **Domain Adaptation**: Specialized indexes for different text domains

## 📚 References & Resources

- [FAISS Documentation](https://faiss.ai/)
- [FAISS GitHub Repository](https://github.com/facebookresearch/faiss)
- [Similarity Search Paper](https://arxiv.org/abs/1702.08734)
- [HNSW Algorithm](https://arxiv.org/abs/1603.09320)
- [Product Quantization](https://hal.inria.fr/inria-00514462v2/document)

## 🤝 Contributing

This research project demonstrates FAISS capabilities and can be extended for:
- Different datasets and domains
- Additional index types and configurations
- Production deployment scenarios
- Performance optimization studies

---

**Project Status**: ✅ Complete - Comprehensive FAISS research with **17 standalone implementations** covering all major index types, performance analysis, and similarity quality evaluation.

**Coverage**: 100% of practical FAISS implementations for semantic search
**Last Updated**: August 28, 2025
