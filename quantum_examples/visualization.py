import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path

class BenchmarkVisualizer:
    """Visualize benchmark results using matplotlib/seaborn."""
    
    def __init__(self, results_file='benchmark_results.json'):
        """Initialize visualizer with results file."""
        self.results_file = results_file
        with open(results_file, 'r') as f:
            self.results = json.load(f)
            
        # Extract param sizes and metrics
        self.param_sizes = [r['paramSize'] for r in self.results]
        
        # Set style for publication-quality plots
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("paper", font_scale=1.5)
        
    def plot_time_scaling(self, save_path=None):
        """Plot computation time scaling."""
        plt.figure(figsize=(10, 6))
        
        times = {
            'Classical': [r['classicalTime'] for r in self.results],
            'Quantum (Simulated)': [r.get('quantumTime', np.nan) for r in self.results],
            'Hybrid': [r['hybridTime'] for r in self.results]
        }
        
        for label, data in times.items():
            plt.plot(self.param_sizes, data, 'o-', label=label, markersize=8)
            
        plt.xlabel('Number of Parameters')
        plt.ylabel('Time (seconds)')
        plt.title('Computation Time Scaling')
        plt.legend()
        plt.yscale('log')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
        
    def plot_loss_comparison(self, save_path=None):
        """Plot loss values comparison."""
        plt.figure(figsize=(10, 6))
        
        losses = {
            'Classical': [r['classicalLoss'] for r in self.results],
            'Quantum': [r.get('quantumLoss', np.nan) for r in self.results],
            'Hybrid': [r['hybridLoss'] for r in self.results]
        }
        
        for label, data in losses.items():
            plt.plot(self.param_sizes, data, 'o-', label=label, markersize=8)
            
        plt.xlabel('Number of Parameters')
        plt.ylabel('Loss Value')
        plt.title('Final Loss Comparison')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
        
    def plot_resource_requirements(self, save_path=None):
        """Plot quantum resource requirements."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Physical qubits
        qubits = [r.get('physicalQubits', np.nan) for r in self.results]
        ax1.plot(self.param_sizes, qubits, 'o-', color='blue', markersize=8)
        ax1.set_xlabel('Number of Parameters')
        ax1.set_ylabel('Physical Qubits Required')
        ax1.set_title('Physical Qubit Scaling')
        ax1.grid(True)
        
        # Embedding time
        embed_times = [r.get('embeddingTime', np.nan) for r in self.results]
        ax2.plot(self.param_sizes, embed_times, 'o-', color='green', markersize=8)
        ax2.set_xlabel('Number of Parameters')
        ax2.set_ylabel('Embedding Time (seconds)')
        ax2.set_title('Embedding Time Scaling')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
        
    def generate_report(self, output_dir='benchmark_plots'):
        """Generate all plots and save them."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self.plot_time_scaling(f'{output_dir}/time_scaling.png')
        self.plot_loss_comparison(f'{output_dir}/loss_comparison.png')
        self.plot_resource_requirements(f'{output_dir}/resource_requirements.png')
        
        # Generate summary statistics
        summary = {
            'max_speedup': self._calculate_max_speedup(),
            'qubit_efficiency': self._calculate_qubit_efficiency(),
            'loss_comparison': self._analyze_loss_patterns()
        }
        
        with open(f'{output_dir}/summary_stats.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
    def _calculate_max_speedup(self):
        """Calculate maximum speedup achieved."""
        speedups = []
        for r in self.results:
            if 'quantumTime' in r:
                speedup = r['classicalTime'] / r['quantumTime']
                speedups.append(speedup)
        return max(speedups) if speedups else 0
        
    def _calculate_qubit_efficiency(self):
        """Calculate qubit efficiency metrics."""
        return {
            'qubits_per_param': [
                r.get('physicalQubits', 0) / r['paramSize'] 
                for r in self.results if 'physicalQubits' in r
            ]
        }
        
    def _analyze_loss_patterns(self):
        """Analyze patterns in loss values."""
        return {
            'quantum_vs_classical': [
                r.get('quantumLoss', 0) / r['classicalLoss']
                for r in self.results if 'quantumLoss' in r
            ]
        }