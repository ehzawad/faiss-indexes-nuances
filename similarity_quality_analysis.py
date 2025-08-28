#!/usr/bin/env python3
"""
FAISS Similarity Quality Analysis
Compares semantic relevance and similarity matching quality across all implementations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import re
from collections import defaultdict
from pathlib import Path

class SimilarityQualityAnalyzer:
    def __init__(self):
        self.implementations = [
            'standalone_flat_l2.py',
            'standalone_flat_ip.py', 
            'standalone_ivf.py',
            'standalone_hnsw.py',
            'standalone_pq.py',
            'standalone_lsh.py',
            'standalone_gpu_flat.py',
            'standalone_gpu_ivfpq.py',
            'standalone_binary_flat.py',
            'standalone_binary_ivf.py',
            'standalone_idmap.py',
            'standalone_shards.py',
            'standalone_scalar_quantizer.py',
            'standalone_ivf_scalar_quantizer.py',
            'standalone_cascade_search.py',
            'standalone_ensemble_search.py',
            'standalone_adaptive_search.py'
        ]
        
        # Test queries with expected semantic themes
        self.test_queries = {
            "জমি কিনেছি নামজারি করতে চাই": "land_purchase_namjari",
            "নিবন্ধন করতে কি লাগবে": "registration_requirements", 
            "প্রতিনিধি দিয়ে আবেদন করা যায়": "representative_application",
            "নামজারি বাতিল হলে কি করবো": "namjari_rejection_appeal",
            "উত্তরাধিকার সূত্রে নামজারি": "inheritance_namjari"
        }
        
        self.results = {}
        
    def extract_search_results(self, output, query):
        """Extract top 5 search results for a specific query"""
        results = []
        
        # Find the query section
        query_pattern = f"Query \\d+: {re.escape(query)}"
        query_match = re.search(query_pattern, output)
        
        if query_match:
            # Get text after query until next query or end
            start_pos = query_match.end()
            next_query = re.search(r"Query \d+:", output[start_pos:])
            end_pos = start_pos + next_query.start() if next_query else len(output)
            
            query_section = output[start_pos:end_pos]
            
            # Extract top 3 results
            result_pattern = r"(\d+)\.\s*\[.*?\]\s*(.+?)(?=\n\s*\d+\.|$)"
            matches = re.findall(result_pattern, query_section, re.DOTALL)
            
            for rank, text in matches[:3]:
                # Clean up the text
                clean_text = re.sub(r'\s+', ' ', text.strip())
                clean_text = clean_text.replace('...', '').strip()
                results.append({
                    'rank': int(rank),
                    'text': clean_text[:100] + ('...' if len(clean_text) > 100 else '')
                })
        
        return results
    
    def run_implementation_for_similarity(self, impl_file):
        """Run implementation and extract similarity results"""
        print(f"Analyzing similarity quality for {impl_file}...")
        
        try:
            result = subprocess.run(
                ['conda', 'run', '-n', 'faiss-cuda12.1', 'python', impl_file],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                return None
                
            output = result.stdout
            
            # Extract results for each test query
            query_results = {}
            for query, theme in self.test_queries.items():
                results = self.extract_search_results(output, query)
                query_results[query] = {
                    'theme': theme,
                    'results': results
                }
            
            return {
                'name': impl_file.replace('standalone_', '').replace('.py', ''),
                'query_results': query_results,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'name': impl_file.replace('standalone_', '').replace('.py', ''),
                'status': 'error',
                'error': str(e)
            }
    
    def calculate_semantic_relevance_score(self, query, results, theme):
        """Calculate semantic relevance score based on keyword matching"""
        theme_keywords = {
            "land_purchase_namjari": ["জমি", "নামজারি", "কিনেছি", "দলিল", "মিউটেশন", "খারিজ"],
            "registration_requirements": ["নিবন্ধন", "কাগজ", "লাগবে", "দরকার", "প্রয়োজন"],
            "representative_application": ["প্রতিনিধি", "আবেদন", "পক্ষে", "অন্য"],
            "namjari_rejection_appeal": ["বাতিল", "নামঞ্জুর", "আপিল", "রিভিউ", "দুইবার"],
            "inheritance_namjari": ["উত্তরাধিকার", "ওয়ারিশ", "মৃত্যু", "বন্টন", "বাবার"]
        }
        
        keywords = theme_keywords.get(theme, [])
        if not keywords:
            return 0.5  # neutral score
        
        total_score = 0
        for i, result in enumerate(results[:3]):  # Top 3 results
            text = result['text'].lower()
            
            # Count keyword matches
            matches = sum(1 for keyword in keywords if keyword in text)
            keyword_score = matches / len(keywords)
            
            # Weight by rank (rank 1 = weight 3, rank 2 = weight 2, rank 3 = weight 1)
            weight = 4 - result['rank']
            total_score += keyword_score * weight
        
        # Normalize by total possible weight (3+2+1 = 6)
        return total_score / 6
    
    def analyze_all_implementations(self):
        """Analyze similarity quality for all implementations"""
        print("Analyzing similarity matching quality across all implementations...")
        
        for impl in self.implementations:
            if Path(impl).exists():
                result = self.run_implementation_for_similarity(impl)
                if result:
                    self.results[result['name']] = result
        
        return self.results
    
    def calculate_similarity_scores(self):
        """Calculate similarity quality scores for each implementation"""
        similarity_scores = {}
        
        for impl_name, data in self.results.items():
            if data['status'] != 'success':
                continue
                
            impl_scores = {}
            total_score = 0
            
            for query, query_data in data['query_results'].items():
                theme = query_data['theme']
                results = query_data['results']
                
                score = self.calculate_semantic_relevance_score(query, results, theme)
                impl_scores[query] = score
                total_score += score
            
            # Average score across all queries
            avg_score = total_score / len(self.test_queries) if self.test_queries else 0
            
            similarity_scores[impl_name] = {
                'average_score': avg_score,
                'query_scores': impl_scores
            }
        
        return similarity_scores
    
    def create_similarity_comparison_chart(self, similarity_scores):
        """Create similarity quality comparison visualization"""
        
        # Prepare data for plotting
        impl_names = []
        avg_scores = []
        query_score_matrix = []
        
        for impl_name, scores in similarity_scores.items():
            impl_names.append(impl_name.replace('_', '\n'))
            avg_scores.append(scores['average_score'])
            
            query_scores = []
            for query in self.test_queries.keys():
                query_scores.append(scores['query_scores'].get(query, 0))
            query_score_matrix.append(query_scores)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('FAISS Similarity Matching Quality Analysis', fontsize=16, fontweight='bold')
        
        # 1. Average Similarity Quality Scores
        bars = ax1.bar(impl_names, avg_scores, color='lightblue', alpha=0.7)
        ax1.set_title('Average Semantic Relevance Score', fontweight='bold')
        ax1.set_ylabel('Relevance Score (0-1)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, avg_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Add color coding for quality levels
        for i, (bar, score) in enumerate(zip(bars, avg_scores)):
            if score >= 0.8:
                bar.set_color('green')
                bar.set_alpha(0.7)
            elif score >= 0.6:
                bar.set_color('orange') 
                bar.set_alpha(0.7)
            else:
                bar.set_color('red')
                bar.set_alpha(0.7)
        
        # 2. Heatmap of query-specific scores
        query_labels = [f"Q{i+1}" for i in range(len(self.test_queries))]
        
        im = ax2.imshow(np.array(query_score_matrix).T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax2.set_title('Query-Specific Relevance Scores', fontweight='bold')
        ax2.set_xlabel('Implementation')
        ax2.set_ylabel('Test Query')
        ax2.set_xticks(range(len(impl_names)))
        ax2.set_xticklabels(impl_names, rotation=45)
        ax2.set_yticks(range(len(query_labels)))
        ax2.set_yticklabels(query_labels)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Relevance Score')
        
        # Add text annotations
        for i in range(len(impl_names)):
            for j in range(len(query_labels)):
                text = ax2.text(i, j, f'{query_score_matrix[i][j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        plt.savefig('faiss_similarity_quality_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_detailed_similarity_report(self, similarity_scores):
        """Create detailed similarity analysis report"""
        
        print("\n" + "="*100)
        print("DETAILED SIMILARITY MATCHING QUALITY ANALYSIS")
        print("="*100)
        
        # Sort by average score
        sorted_impls = sorted(similarity_scores.items(), 
                            key=lambda x: x[1]['average_score'], 
                            reverse=True)
        
        print(f"{'Rank':<4} {'Algorithm':<20} {'Avg Score':<10} {'Quality':<12} {'Best Query':<15} {'Worst Query':<15}")
        print("-" * 100)
        
        for rank, (impl_name, scores) in enumerate(sorted_impls, 1):
            avg_score = scores['average_score']
            
            # Determine quality level
            if avg_score >= 0.8:
                quality = "Excellent"
            elif avg_score >= 0.6:
                quality = "Good"
            elif avg_score >= 0.4:
                quality = "Fair"
            else:
                quality = "Poor"
            
            # Find best and worst queries
            query_scores = scores['query_scores']
            best_query = max(query_scores.items(), key=lambda x: x[1])
            worst_query = min(query_scores.items(), key=lambda x: x[1])
            
            print(f"{rank:<4} {impl_name:<20} {avg_score:<10.3f} {quality:<12} "
                  f"Q{list(self.test_queries.keys()).index(best_query[0])+1}({best_query[1]:.2f})<15 "
                  f"Q{list(self.test_queries.keys()).index(worst_query[0])+1}({worst_query[1]:.2f})")
        
        print("\n" + "="*100)
        print("QUERY LEGEND:")
        for i, (query, theme) in enumerate(self.test_queries.items(), 1):
            print(f"Q{i}: {query[:50]}... ({theme})")
        
        print("\n" + "="*100)
        print("SIMILARITY QUALITY INSIGHTS:")
        print("="*100)
        
        # Top performers
        top_3 = sorted_impls[:3]
        print(f"\n🏆 TOP SIMILARITY PERFORMERS:")
        for i, (name, scores) in enumerate(top_3, 1):
            print(f"  {i}. {name}: {scores['average_score']:.3f} - Excellent semantic matching")
        
        # Bottom performers  
        bottom_3 = sorted_impls[-3:]
        print(f"\n⚠️  NEEDS IMPROVEMENT:")
        for i, (name, scores) in enumerate(bottom_3, 1):
            print(f"  {i}. {name}: {scores['average_score']:.3f} - Consider parameter tuning")
        
        # Algorithm type analysis
        exact_algos = [name for name, _ in sorted_impls if any(x in name for x in ['flat_l2', 'flat_ip'])]
        approx_algos = [name for name, _ in sorted_impls if any(x in name for x in ['hnsw', 'ivf', 'lsh'])]
        
        if exact_algos:
            exact_avg = np.mean([similarity_scores[name]['average_score'] for name in exact_algos])
            print(f"\n📊 EXACT ALGORITHMS AVG: {exact_avg:.3f}")
        
        if approx_algos:
            approx_avg = np.mean([similarity_scores[name]['average_score'] for name in approx_algos])
            print(f"📊 APPROXIMATE ALGORITHMS AVG: {approx_avg:.3f}")
        
        return sorted_impls
    
    def run_complete_similarity_analysis(self):
        """Run complete similarity quality analysis"""
        print("Starting comprehensive similarity matching quality analysis...")
        
        # Analyze all implementations
        self.analyze_all_implementations()
        
        # Calculate similarity scores
        similarity_scores = self.calculate_similarity_scores()
        
        # Create visualizations
        self.create_similarity_comparison_chart(similarity_scores)
        
        # Create detailed report
        ranking = self.create_detailed_similarity_report(similarity_scores)
        
        # Save results
        results_df = []
        for impl_name, scores in similarity_scores.items():
            row = {'Algorithm': impl_name, 'Average_Score': scores['average_score']}
            for i, query in enumerate(self.test_queries.keys(), 1):
                row[f'Query_{i}_Score'] = scores['query_scores'].get(query, 0)
            results_df.append(row)
        
        df = pd.DataFrame(results_df)
        df = df.sort_values('Average_Score', ascending=False)
        df.to_csv('faiss_similarity_quality_scores.csv', index=False)
        
        print(f"\n✅ Similarity analysis complete!")
        print(f"📊 Visualization saved: faiss_similarity_quality_analysis.png")
        print(f"📄 Detailed scores saved: faiss_similarity_quality_scores.csv")
        
        return similarity_scores

def main():
    analyzer = SimilarityQualityAnalyzer()
    analyzer.run_complete_similarity_analysis()

if __name__ == "__main__":
    main()
