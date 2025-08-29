#!/usr/bin/env python3
"""
Focused FAISS Visualizations - Multiple Clear Charts
Creates separate, focused visualizations instead of overwhelming single chart
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class FocusedFAISSVisualizer:
    def __init__(self):
        # Performance data from actual runs with 910 Bengali Q&A samples
        self.performance_data = {
            'FlatL2': {'build_ms': 0.3, 'search_ms': 0.17, 'memory_mb': 1.33, 'accuracy': 100, 'compression': 1},
            'FlatIP': {'build_ms': 0.4, 'search_ms': 0.11, 'memory_mb': 1.33, 'accuracy': 100, 'compression': 1},
            'HNSW': {'build_ms': 4.7, 'search_ms': 0.07, 'memory_mb': 1.60, 'accuracy': 92, 'compression': 1},
            'IVF': {'build_ms': 7.2, 'search_ms': 0.07, 'memory_mb': 1.33, 'accuracy': 95, 'compression': 1},
            'PQ': {'build_ms': 207.6, 'search_ms': 0.74, 'memory_mb': 0.01, 'accuracy': 80, 'compression': 192},
            'LSH': {'build_ms': 12.7, 'search_ms': 0.24, 'memory_mb': 0.03, 'accuracy': 80, 'compression': 1},
            'Binary Flat': {'build_ms': 0.9, 'search_ms': 0.21, 'memory_mb': 0.042, 'accuracy': 75, 'compression': 32},
            'Scalar Quantizer': {'build_ms': 1.3, 'search_ms': 0.32, 'memory_mb': 0.33, 'accuracy': 87, 'compression': 4},
            'IDMap': {'build_ms': 0.9, 'search_ms': 0.16, 'memory_mb': 1.33, 'accuracy': 100, 'compression': 1},
            'Ensemble': {'build_ms': 12.4, 'search_ms': 0.22, 'memory_mb': 2.5, 'accuracy': 90, 'compression': 1}
        }
        
        self.categories = {
            'Exact': ['FlatL2', 'FlatIP', 'IDMap'],
            'Approximate': ['HNSW', 'IVF'],
            'Compressed': ['PQ', 'Scalar Quantizer', 'Binary Flat'],
            'Hash-based': ['LSH'],
            'Hybrid': ['Ensemble']
        }
        
        self.colors = {
            'Exact': '#FF6B6B',
            'Approximate': '#4ECDC4', 
            'Compressed': '#45B7D1',
            'Hash-based': '#96CEB4',
            'Hybrid': '#FFEAA7'
        }
    
    def create_speed_comparison(self):
        """Chart 1: Search Speed Comparison"""
        plt.figure(figsize=(12, 8))
        
        algorithms = list(self.performance_data.keys())
        search_times = [self.performance_data[alg]['search_ms'] for alg in algorithms]
        
        # Get colors for each algorithm
        alg_colors = []
        for alg in algorithms:
            for category, algs in self.categories.items():
                if alg in algs:
                    alg_colors.append(self.colors[category])
                    break
        
        bars = plt.bar(range(len(algorithms)), search_times, color=alg_colors, alpha=0.8, edgecolor='black')
        
        plt.title('FAISS Search Speed Comparison\n910 Bengali Q&A Samples', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Search Time (milliseconds)', fontsize=12)
        plt.xlabel('Algorithm', fontsize=12)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, search_times)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(search_times)*0.02,
                    f'{value:.2f}ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.xticks(range(len(algorithms)), algorithms, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, label=category) 
                          for category, color in self.colors.items()]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.savefig('faiss_speed_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_memory_efficiency(self):
        """Chart 2: Memory Usage and Compression"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        algorithms = list(self.performance_data.keys())
        memory_usage = [self.performance_data[alg]['memory_mb'] for alg in algorithms]
        compression_ratios = [self.performance_data[alg]['compression'] for alg in algorithms]
        
        # Get colors
        alg_colors = []
        for alg in algorithms:
            for category, algs in self.categories.items():
                if alg in algs:
                    alg_colors.append(self.colors[category])
                    break
        
        # Memory Usage Chart
        bars1 = ax1.bar(range(len(algorithms)), memory_usage, color=alg_colors, alpha=0.8, edgecolor='black')
        ax1.set_title('Memory Usage Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Memory Usage (MB)', fontsize=12)
        ax1.set_yscale('log')
        ax1.set_xticks(range(len(algorithms)))
        ax1.set_xticklabels(algorithms, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, value in zip(bars1, memory_usage):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.2,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Compression Ratio Chart
        bars2 = ax2.bar(range(len(algorithms)), compression_ratios, color=alg_colors, alpha=0.8, edgecolor='black')
        ax2.set_title('Compression Ratio Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Compression Ratio (x)', fontsize=12)
        ax2.set_yscale('log')
        ax2.set_xticks(range(len(algorithms)))
        ax2.set_xticklabels(algorithms, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, value in zip(bars2, compression_ratios):
            if value > 1:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.2,
                        f'{value}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.suptitle('FAISS Memory Efficiency Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('faiss_memory_efficiency.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_accuracy_analysis(self):
        """Chart 3: Accuracy vs Speed Trade-off"""
        plt.figure(figsize=(12, 8))
        
        algorithms = list(self.performance_data.keys())
        search_times = [self.performance_data[alg]['search_ms'] for alg in algorithms]
        accuracies = [self.performance_data[alg]['accuracy'] for alg in algorithms]
        
        # Get colors and sizes based on compression
        alg_colors = []
        sizes = []
        for alg in algorithms:
            for category, algs in self.categories.items():
                if alg in algs:
                    alg_colors.append(self.colors[category])
                    break
            # Size based on compression (inverse - higher compression = smaller size)
            compression = self.performance_data[alg]['compression']
            sizes.append(max(50, 500 / compression))
        
        scatter = plt.scatter(search_times, accuracies, c=alg_colors, s=sizes, alpha=0.7, edgecolors='black', linewidth=2)
        
        # Add algorithm labels
        for i, alg in enumerate(algorithms):
            plt.annotate(alg, (search_times[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
        
        plt.xlabel('Search Time (milliseconds)', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('FAISS Accuracy vs Speed Trade-off\n(Bubble size = inverse compression ratio)', fontsize=16, fontweight='bold', pad=20)
        plt.grid(alpha=0.3)
        
        # Add legend for categories
        legend_elements = [plt.scatter([], [], c=color, s=100, alpha=0.7, edgecolors='black', label=category) 
                          for category, color in self.colors.items()]
        plt.legend(handles=legend_elements, loc='lower left')
        
        plt.tight_layout()
        plt.savefig('faiss_accuracy_speed_tradeoff.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_build_time_analysis(self):
        """Chart 4: Build Time Analysis"""
        plt.figure(figsize=(12, 8))
        
        algorithms = list(self.performance_data.keys())
        build_times = [self.performance_data[alg]['build_ms'] for alg in algorithms]
        
        # Get colors
        alg_colors = []
        for alg in algorithms:
            for category, algs in self.categories.items():
                if alg in algs:
                    alg_colors.append(self.colors[category])
                    break
        
        bars = plt.bar(range(len(algorithms)), build_times, color=alg_colors, alpha=0.8, edgecolor='black')
        
        plt.title('FAISS Build Time Comparison\n910 Bengali Q&A Samples', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Build Time (milliseconds)', fontsize=12)
        plt.xlabel('Algorithm', fontsize=12)
        plt.yscale('log')
        
        # Add value labels
        for bar, value in zip(bars, build_times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.2,
                    f'{value:.1f}ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.xticks(range(len(algorithms)), algorithms, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, label=category) 
                          for category, color in self.colors.items()]
        plt.legend(handles=legend_elements, loc='upper left')
        
        plt.tight_layout()
        plt.savefig('faiss_build_time_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_performance_radar(self):
        """Chart 5: Performance Radar for Top Algorithms"""
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Select top 5 algorithms
        top_algorithms = ['HNSW', 'IVF', 'FlatIP', 'PQ', 'Binary Flat']
        metrics = ['Speed\n(inverse ms)', 'Memory Efficiency\n(inverse MB)', 'Build Speed\n(inverse ms)', 'Accuracy\n(%)', 'Compression\n(ratio)']
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, alg in enumerate(top_algorithms):
            if alg in self.performance_data:
                data = self.performance_data[alg]
                
                # Normalize metrics (higher is better)
                speed_score = 1 / (data['search_ms'] + 0.01) * 10
                memory_score = 1 / (data['memory_mb'] + 0.01) * 10
                build_score = 1 / (data['build_ms'] + 0.01) * 100
                accuracy_score = data['accuracy']
                compression_score = min(data['compression'], 50)  # Cap for visualization
                
                values = [speed_score, memory_score, build_score, accuracy_score, compression_score]
                values += values[:1]  # Complete the circle
                
                # Get color for algorithm
                color = '#FF6B6B'  # Default
                for category, algs in self.categories.items():
                    if alg in algs:
                        color = self.colors[category]
                        break
                
                ax.plot(angles, values, 'o-', linewidth=3, label=alg, color=color, markersize=8)
                ax.fill(angles, values, alpha=0.1, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_title('FAISS Algorithm Performance Radar\nTop 5 Algorithms Comparison', 
                     fontsize=14, fontweight='bold', pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=12)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('faiss_performance_radar.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_use_case_matrix(self):
        """Chart 6: Use Case Recommendation Matrix"""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create use case matrix
        use_cases = ['Small Dataset\n(<1K)', 'Medium Dataset\n(1K-10K)', 'Large Dataset\n(>10K)', 
                    'Memory Critical', 'Speed Critical', 'Accuracy Critical', 'Production Ready']
        
        algorithms = ['FlatL2', 'FlatIP', 'HNSW', 'IVF', 'PQ', 'Binary Flat', 'LSH', 'IDMap']
        
        # Recommendation matrix (0-3 scale: 0=Poor, 1=Fair, 2=Good, 3=Excellent)
        recommendations = np.array([
            [3, 3, 1, 0, 1, 3, 2],  # FlatL2
            [3, 3, 1, 0, 2, 3, 2],  # FlatIP
            [2, 3, 3, 1, 3, 2, 3],  # HNSW
            [2, 3, 3, 2, 3, 2, 3],  # IVF
            [1, 2, 3, 3, 1, 1, 2],  # PQ
            [1, 2, 3, 3, 2, 1, 2],  # Binary Flat
            [1, 2, 2, 2, 2, 1, 1],  # LSH
            [2, 2, 2, 1, 2, 3, 3],  # IDMap
        ])
        
        # Create heatmap
        im = ax.imshow(recommendations, cmap='RdYlGn', aspect='auto', vmin=0, vmax=3)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(use_cases)))
        ax.set_yticks(np.arange(len(algorithms)))
        ax.set_xticklabels(use_cases, fontsize=11)
        ax.set_yticklabels(algorithms, fontsize=11)
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(len(algorithms)):
            for j in range(len(use_cases)):
                value = recommendations[i, j]
                text_color = 'white' if value < 1.5 else 'black'
                rating = ['Poor', 'Fair', 'Good', 'Excellent'][value]
                ax.text(j, i, rating, ha="center", va="center", color=text_color, fontweight='bold')
        
        ax.set_title("FAISS Algorithm Use Case Recommendation Matrix", fontsize=16, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Recommendation Level', rotation=-90, va="bottom", fontsize=12)
        
        plt.tight_layout()
        plt.savefig('faiss_use_case_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_all_visualizations(self):
        """Generate all focused visualizations"""
        print("🎨 Generating focused FAISS visualizations...")
        
        print("📊 Chart 1: Speed Comparison")
        self.create_speed_comparison()
        
        print("💾 Chart 2: Memory Efficiency")
        self.create_memory_efficiency()
        
        print("🎯 Chart 3: Accuracy vs Speed")
        self.create_accuracy_analysis()
        
        print("🏗️ Chart 4: Build Time Analysis")
        self.create_build_time_analysis()
        
        print("🌟 Chart 5: Performance Radar")
        self.create_performance_radar()
        
        print("📋 Chart 6: Use Case Matrix")
        self.create_use_case_matrix()
        
        print("\n✅ All visualizations generated!")
        print("📁 Files created:")
        print("  - faiss_speed_comparison.png")
        print("  - faiss_memory_efficiency.png") 
        print("  - faiss_accuracy_speed_tradeoff.png")
        print("  - faiss_build_time_comparison.png")
        print("  - faiss_performance_radar.png")
        print("  - faiss_use_case_matrix.png")

def main():
    visualizer = FocusedFAISSVisualizer()
    visualizer.generate_all_visualizations()

if __name__ == "__main__":
    main()
