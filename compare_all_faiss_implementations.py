#!/usr/bin/env python3
"""
Comprehensive FAISS Implementation Comparison
Runs all 17 standalone implementations and creates comparison graphs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import json
import time
import re
from pathlib import Path

class FAISSComparisonAnalyzer:
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
        self.results = {}
        self.test_query = "জমি কিনেছি নামজারি করতে চাই"
        
    def run_single_implementation(self, impl_file):
        """Run a single implementation and extract metrics"""
        print(f"Running {impl_file}...")
        
        try:
            # Run the implementation
            result = subprocess.run(
                ['conda', 'run', '-n', 'faiss-cuda12.1', 'python', impl_file],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            output = result.stdout
            
            # Extract metrics using regex
            metrics = {
                'name': impl_file.replace('standalone_', '').replace('.py', ''),
                'build_time': 0,
                'search_time': 0,
                'accuracy': 'N/A',
                'memory_mb': 0,
                'compression': '1x',
                'top_result': '',
                'status': 'success' if result.returncode == 0 else 'failed'
            }
            
            # Parse build time
            build_match = re.search(r'Build time: ([\d.]+)s', output)
            if build_match:
                metrics['build_time'] = float(build_match.group(1))
            
            # Parse search time
            search_match = re.search(r'Average search time: ([\d.]+)ms', output)
            if search_match:
                metrics['search_time'] = float(search_match.group(1))
            
            # Parse accuracy
            acc_match = re.search(r'Accuracy: ([^\\n]+)', output)
            if acc_match:
                metrics['accuracy'] = acc_match.group(1).strip()
            
            # Parse memory usage
            mem_match = re.search(r'Memory usage: ([\d.]+)MB', output)
            if mem_match:
                metrics['memory_mb'] = float(mem_match.group(1))
            
            # Parse compression
            comp_match = re.search(r'Compression: ([\d.]+)x', output)
            if comp_match:
                metrics['compression'] = comp_match.group(1) + 'x'
            
            # Get first search result for the test query
            query_section = output.split(f'Query 1: {self.test_query}')
            if len(query_section) > 1:
                result_match = re.search(r'1\. \[.*?\] (.{0,60})', query_section[1])
                if result_match:
                    metrics['top_result'] = result_match.group(1).strip() + '...'
            
            return metrics
            
        except subprocess.TimeoutExpired:
            return {
                'name': impl_file.replace('standalone_', '').replace('.py', ''),
                'status': 'timeout',
                'build_time': 0,
                'search_time': 0,
                'accuracy': 'N/A',
                'memory_mb': 0,
                'compression': '1x',
                'top_result': 'Timeout'
            }
        except Exception as e:
            return {
                'name': impl_file.replace('standalone_', '').replace('.py', ''),
                'status': 'error',
                'build_time': 0,
                'search_time': 0,
                'accuracy': 'N/A',
                'memory_mb': 0,
                'compression': '1x',
                'top_result': f'Error: {str(e)}'
            }
    
    def run_all_implementations(self):
        """Run all implementations and collect results"""
        print("Running all 17 FAISS implementations...")
        
        for impl in self.implementations:
            if Path(impl).exists():
                metrics = self.run_single_implementation(impl)
                self.results[metrics['name']] = metrics
            else:
                print(f"Warning: {impl} not found")
        
        return self.results
    
    def create_performance_comparison(self):
        """Create performance comparison charts"""
        if not self.results:
            print("No results to plot")
            return
        
        # Prepare data
        names = []
        build_times = []
        search_times = []
        memory_usage = []
        statuses = []
        
        for name, metrics in self.results.items():
            if metrics['status'] == 'success':
                names.append(name.replace('_', '\n'))
                build_times.append(metrics['build_time'] * 1000)  # Convert to ms
                search_times.append(metrics['search_time'])
                memory_usage.append(metrics['memory_mb'])
                statuses.append('success')
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('FAISS Implementation Comparison - All 17 Algorithms', fontsize=16, fontweight='bold')
        
        # 1. Build Time Comparison
        bars1 = ax1.bar(names, build_times, color='skyblue', alpha=0.7)
        ax1.set_title('Build Time Comparison', fontweight='bold')
        ax1.set_ylabel('Build Time (ms)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, build_times):
            if value > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(build_times)*0.01,
                        f'{value:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Search Time Comparison
        bars2 = ax2.bar(names, search_times, color='lightcoral', alpha=0.7)
        ax2.set_title('Average Search Time Comparison', fontweight='bold')
        ax2.set_ylabel('Search Time (ms)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, value in zip(bars2, search_times):
            if value > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(search_times)*0.01,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 3. Memory Usage Comparison
        bars3 = ax3.bar(names, memory_usage, color='lightgreen', alpha=0.7)
        ax3.set_title('Memory Usage Comparison', fontweight='bold')
        ax3.set_ylabel('Memory Usage (MB)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(axis='y', alpha=0.3)
        
        for bar, value in zip(bars3, memory_usage):
            if value > 0:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(memory_usage)*0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 4. Speed vs Memory Scatter Plot
        ax4.scatter(search_times, memory_usage, s=100, alpha=0.7, c='purple')
        for i, name in enumerate(names):
            ax4.annotate(name.replace('\n', '_'), (search_times[i], memory_usage[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax4.set_xlabel('Search Time (ms)')
        ax4.set_ylabel('Memory Usage (MB)')
        ax4.set_title('Speed vs Memory Tradeoff', fontweight='bold')
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('faiss_implementations_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_results_table(self):
        """Create detailed results comparison table"""
        if not self.results:
            return None
        
        # Create DataFrame
        df_data = []
        for name, metrics in self.results.items():
            df_data.append({
                'Algorithm': name.replace('_', ' ').title(),
                'Build Time (ms)': f"{metrics['build_time']*1000:.1f}" if metrics['build_time'] > 0 else 'N/A',
                'Search Time (ms)': f"{metrics['search_time']:.2f}" if metrics['search_time'] > 0 else 'N/A',
                'Memory (MB)': f"{metrics['memory_mb']:.3f}" if metrics['memory_mb'] > 0 else 'N/A',
                'Accuracy': metrics['accuracy'],
                'Compression': metrics['compression'],
                'Status': metrics['status'],
                'Top Result': metrics['top_result'][:50] + '...' if len(metrics['top_result']) > 50 else metrics['top_result']
            })
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('Search Time (ms)', key=lambda x: pd.to_numeric(x.str.replace('N/A', '999'), errors='coerce'))
        
        print("\n" + "="*120)
        print("COMPREHENSIVE FAISS IMPLEMENTATION COMPARISON")
        print("="*120)
        print(df.to_string(index=False))
        print("="*120)
        
        # Save to CSV
        df.to_csv('faiss_comparison_results.csv', index=False)
        
        return df
    
    def analyze_inference_consistency(self):
        """Analyze if all implementations return consistent results for the same query"""
        print(f"\nInference Consistency Analysis for query: '{self.test_query}'")
        print("-" * 80)
        
        successful_results = {}
        for name, metrics in self.results.items():
            if metrics['status'] == 'success' and metrics['top_result']:
                successful_results[name] = metrics['top_result']
        
        # Group similar results
        result_groups = {}
        for name, result in successful_results.items():
            # Simple similarity check based on first few words
            key_words = ' '.join(result.split()[:5])
            if key_words not in result_groups:
                result_groups[key_words] = []
            result_groups[key_words].append(name)
        
        print(f"Found {len(result_groups)} distinct result groups:")
        for i, (key, algorithms) in enumerate(result_groups.items(), 1):
            print(f"\nGroup {i} ({len(algorithms)} algorithms):")
            print(f"  Result: {key}...")
            print(f"  Algorithms: {', '.join(algorithms)}")
        
        return result_groups
    
    def run_complete_analysis(self):
        """Run complete comparison analysis"""
        print("Starting comprehensive FAISS implementation analysis...")
        
        # Run all implementations
        self.run_all_implementations()
        
        # Create visualizations
        self.create_performance_comparison()
        
        # Create results table
        self.create_results_table()
        
        # Analyze inference consistency
        self.analyze_inference_consistency()
        
        # Summary statistics
        successful = sum(1 for r in self.results.values() if r['status'] == 'success')
        failed = len(self.results) - successful
        
        print(f"\n{'='*60}")
        print("ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Total implementations: {len(self.results)}")
        print(f"Successful runs: {successful}")
        print(f"Failed runs: {failed}")
        
        if successful > 0:
            build_times = [r['build_time']*1000 for r in self.results.values() if r['status'] == 'success' and r['build_time'] > 0]
            search_times = [r['search_time'] for r in self.results.values() if r['status'] == 'success' and r['search_time'] > 0]
            
            if build_times:
                print(f"Build time range: {min(build_times):.1f} - {max(build_times):.1f} ms")
            if search_times:
                print(f"Search time range: {min(search_times):.2f} - {max(search_times):.2f} ms")
        
        print(f"Results saved to: faiss_comparison_results.csv")
        print(f"Visualization saved to: faiss_implementations_comparison.png")

def main():
    analyzer = FAISSComparisonAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
