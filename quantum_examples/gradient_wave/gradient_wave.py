from typing import List, Tuple, Optional
import minorminer
import numpy as np
import dimod
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
import time
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress

class GradientWave:
    """
    Demonstrates using D-Wave quantum annealing for optimizing model parameters
    by mapping gradient descent to a QUBO problem.
    """
    
    def __init__(self, backend_type: str = "simulator", shots: int = 100):
        """
        Initialize the Gradient Wave optimizer.
        
        Args:
            backend_type (str): Type of quantum backend to use ("simulator", "hardware", or "hybrid")
            shots (int): Number of shots for quantum circuit execution
        """
        self.console = Console()
        self.backend_type = backend_type
        self.shots = shots
        self.config = {
            "iterations": 40,
            "restarts": 2
        }
        
        # Initialize D-Wave sampler
        if backend_type == "hardware":
            self.sampler = EmbeddingComposite(DWaveSampler())
            self.console.print("[yellow]Warning: Using real D-Wave quantum hardware!")
        elif backend_type == "hybrid":
            self.sampler = LeapHybridSampler()
            self.console.print("[yellow]Using D-Wave hybrid solver")
        else:
            self.sampler = dimod.SimulatedAnnealingSampler()
    
    def create_loss_landscape(self, complexity: str = "simple") -> callable:
        """
        Create computationally intensive loss landscapes.
        
        Args:
            complexity (str): Complexity of the loss landscape ("simple", "medium", "complex")
            
        Returns:
            callable: Loss function that takes parameters and returns loss value
        """
        if complexity == "simple":
            def simple_loss(params):
                # O(n²) complexity with matrix operations
                param_matrix = np.outer(params, params)
                result = np.sum(param_matrix ** 2)
                
                # Add periodic barriers to create local minima
                for i in range(len(params)):
                    result += np.sin(2 * np.pi * params[i]) * np.cos(3 * np.pi * params[i])
                    
                # Add cross-terms for all pairs
                for i in range(len(params)):
                    for j in range(i + 1, len(params)):
                        result += 0.1 * np.sin(params[i] + params[j])
                
                return float(result)
            return simple_loss
            
        elif complexity == "medium":
            def medium_loss(params):
                # O(n³) complexity with tensor operations
                param_tensor = np.einsum('i,j,k->ijk', params, params, params)
                result = np.sum(param_tensor ** 2)
                
                # Multiple frequency components for each parameter
                for i in range(len(params)):
                    for freq in range(1, 6):
                        result += np.sin(freq * params[i]) * np.cos((freq + 1) * params[i])
                
                # Triple parameter interactions
                for i in range(len(params)):
                    for j in range(i + 1, len(params)):
                        for k in range(j + 1, len(params)):
                            result += params[i] * params[j] * params[k]
                            result += np.sin(params[i] + params[j] + params[k])
                
                return float(result)
            return medium_loss
            
        else:
            def complex_loss(params):
                # O(n⁴) complexity with hypercube operations
                n = len(params)
                result = 0.0
                
                # Fourth-order interactions
                for i in range(n):
                    for j in range(i + 1, n):
                        for k in range(j + 1, n):
                            for l in range(k + 1, n):
                                term = params[i] * params[j] * params[k] * params[l]
                                result += term ** 2
                                result += np.sin(term)
                
                # Multiple frequency components with cross-terms
                for i in range(n):
                    for j in range(i + 1, n):
                        for freq in range(1, 8):
                            result += np.sin(freq * params[i] + freq * params[j])
                            result += np.cos((freq + 1) * params[i] - freq * params[j])
                
                return float(result)
            return complex_loss
    
    def calculate_gradient(
        self, loss_fn: callable,
        params: np.ndarray,
        epsilon: float = 1e-7
    ) -> np.ndarray:
        """
        Calculate numerical gradient of the loss function with scaling.
        
        Args:
            loss_fn (callable): Loss function
            params (np.ndarray): Current parameters
            epsilon (float): Small value for numerical gradient calculation
            
        Returns:
            np.ndarray: Gradient vector
        """
        gradient = np.zeros_like(params, dtype=np.float64)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += epsilon
            params_minus = params.copy()
            params_minus[i] -= epsilon
            
            # Calculate gradient with scaling to avoid overflow
            loss_diff = loss_fn(params_plus) - loss_fn(params_minus)
            gradient[i] = (loss_diff / (2 * epsilon))
            
            # Apply scaling to keep values in reasonable range
            gradient[i] = np.clip(gradient[i], -1e3, 1e3)
            # Normalize gradients to prevent explosion
            gradient_norm = np.linalg.norm(gradient)
            if gradient_norm > 0:
                gradient = gradient / gradient_norm
            
        return gradient
    
    def construct_qubo(
        self, gradient: np.ndarray,
        current_params: np.ndarray,
        learning_rate: float = 0.1
    ) -> dict:
        """
        Construct QUBO matrix for gradient descent step optimization.
        
        Args:
            gradient (np.ndarray): Gradient vector
            current_params (np.ndarray): Current parameter values
            learning_rate (float): Learning rate for gradient descent
            
        Returns:
            dict: QUBO dictionary
        """
        num_params = len(gradient)
        qubo = {}
        
        # Convert gradient descent step to QUBO
        # We want to minimize: ||params - (current_params - lr * gradient)||^2
        for i in range(num_params):
            # Diagonal terms
            qubo[(f'p{i}', f'p{i}')] = 1.0
            
            # Linear terms
            target = current_params[i] - learning_rate * gradient[i]
            qubo[(f'p{i}', f'p{i}')] += -2.0 * target
            
            # Interaction terms
            for j in range(i+1, num_params):
                qubo[(f'p{i}', f'p{j}')] = 2.0
        
        self.qubo = qubo
        return qubo
    
    def check_embedding(self) -> bool:
        """
        Check if problem can be embedded on the hardware.
        
        Args:
            qubo: QUBO dictionary
            
        Returns:
            bool: True if embedding is possible
        """
        # if self.backend_type == "simulator" or self.backend_type == "hybrid":
        #     return True
            
        try:
            # Get the hardware graph
            dw_sampler = DWaveSampler()
            target_graph = dw_sampler.edgelist
            
            # Try to find embedding
            embedding = minorminer.find_embedding(self.qubo.keys(), target_graph)
            
            if embedding:
                num_physical_qubits = sum(len(chain) for chain in embedding.values())
                self.console.print(f"[green]Found valid embedding using {num_physical_qubits} physical qubits")
                return True
            else:
                self.console.print("[red]No valid embedding found")
                return False
                
        except Exception as e:
            self.console.print(f"[red]Error checking embedding: {str(e)}")
            return False
    
    def decode_solution(self, sample: dict, num_params: int) -> np.ndarray:
        """
        Decode QUBO solution back to parameter space.
        
        Args:
            sample (dict): Solution from quantum annealer
            num_params (int): Number of parameters
            
        Returns:
            np.ndarray: Optimized parameters
        """
        # Handle both dictionary and dimod.SampleView formats
        if hasattr(sample, 'get'):
            params = np.array([sample.get(f'p{i}', 0.0) for i in range(num_params)])
        else:
            sample_dict = dict(sample)
            params = np.array([sample_dict.get(f'p{i}', 0.0) for i in range(num_params)])
        
        # Add small noise to prevent perfect solutions
        noise = np.random.normal(0, 0.001, params.shape)
        return params + noise
        
    
    def optimize(
        self,
        num_params: int = 2,
        max_steps: int = 10,
        complexity: str = "simple"
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Run quantum-assisted gradient descent optimization.
        
        Args:
            num_params (int): Number of parameters to optimize
            max_steps (int): Maximum optimization steps
            complexity (str): Complexity of loss landscape
            
        Returns:
            Tuple[List[np.ndarray], List[float]]: Parameter history and loss history
        """
        loss_fn = self.create_loss_landscape(complexity)
        # Initialize with small random values
        params = np.random.randn(num_params).astype(np.float64) * 0.1
        param_history = [params.copy()]
        loss_history = [float(loss_fn(params))]
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Optimizing with quantum annealing...", total=max_steps)
            
            for step in range(max_steps):
                # Calculate gradient
                gradient = self.calculate_gradient(loss_fn, params)
                
                # Construct QUBO for this step
                qubo = self.construct_qubo(gradient, params)
                
                # Run quantum annealing
                response = self.sampler.sample_qubo(qubo, num_reads=100)
                
                # Get best solution
                best_sample = next(response.samples())
                new_params = self.decode_solution(best_sample, num_params)
                
                # Update parameters
                params = new_params
                param_history.append(params.copy())
                current_loss = float(loss_fn(params))
                loss_history.append(current_loss)
                
                # Update progress
                progress.update(task, advance=1)
                
                # Early stopping if converged
                if step > 0 and abs(loss_history[-1] - loss_history[-2]) < 1e-6:
                    progress.update(task, completed=max_steps)
                    break
                    
            # Ensure progress shows complete
            progress.update(task, completed=max_steps)
        
        return param_history, loss_history

    def _print_comparison(
        self,
        classical_loss: float,
        quantum_loss: float, 
        classical_time: float,
        total_time: float
    ):
        """Print comparison between classical and quantum approaches."""
        abs_diff = abs(abs(quantum_loss) - abs(classical_loss))
        abs_base = abs(classical_loss)
        percentage = (abs_diff / abs_base) * 100

        if classical_loss == 0 or quantum_loss == 0:
            self.console.print("[yellow]Warning: Perfect zero loss achieved - results may be unrealistic")
            
        if abs(quantum_loss) < abs(classical_loss):
            self.console.print(f"\n[green]Quantum annealing found a {percentage:.1f}% better minimum")
        else:
            self.console.print(f"\n[yellow]Classical approach found a {percentage:.1f}% better minimum")

        time_ratio = total_time / classical_time
        self.console.print(f"Time comparison: Quantum took {time_ratio:.1f}x longer than classical")

    def estimate_hardware_time(self, num_problems: int, shots_per_problem: int = 100):
        """
        Estimate D-Wave hardware time required.
        
        Args:
            num_problems: Number of QUBO problems to solve
            shots_per_problem: Number of annealing shots per problem
        """
        # D-Wave constants (from documentation)
        ANNEALING_TIME_US = 20  # microseconds
        READOUT_TIME_US = 100   # microseconds per shot
        PROGRAMMING_TIME_US = 10000  # microseconds per problem
        
        total_time_us = num_problems * (
            PROGRAMMING_TIME_US +
            shots_per_problem * (ANNEALING_TIME_US + READOUT_TIME_US)
        )
        
        total_time_s = total_time_us / 1e6
        self.console.print(f"\n[cyan]Hardware time estimate:")
        self.console.print(f"Total problems: {num_problems}")
        self.console.print(f"Shots per problem: {shots_per_problem}")
        self.console.print(f"Estimated QPU time: {total_time_s:.2f} seconds")
        self.console.print(f"With typical queue times this could take {total_time_s * 10:.0f}-{total_time_s * 30:.0f} seconds")

    def run_demo(self, dry_run: bool = False):
        """Run an interactive demo comparing classical and quantum optimization."""
        dry_run_compute = 0
        self.console.print(Panel.fit(
            "[bold cyan]Welcome to Gradient Wave![/bold cyan]\n"
            "Using D-Wave quantum annealing to optimize neural network parameters"
        ))
        
        # Configuration for different complexity levels
        configs = {
            "simple": {"params": 20, "restarts": 2, "iterations": 40},
            "medium": {"params": 20, "restarts": 3, "iterations": 40},
            "complex": {"params": 20, "restarts": 4, "iterations": 40}
        }
        
        for complexity in ["simple", "medium", "complex"]:
            self.console.print(f"\n[cyan]Testing {complexity} loss landscape:")
            config = configs[complexity]
            
            # Classical gradient descent with multiple starting points
            classical_start = time.time()
            best_classical_loss = float('inf')
            self.console.print(f"\nComplexity: {complexity} (O(n^{2 if complexity == 'simple' else 3 if complexity == 'medium' else 4}))")
            self.console.print(f"Classical iterations: {config['iterations'] * config['restarts']}")
            self.console.print(f"Quantum shots per iteration: {self.shots}")  # Add actual shot count
            
            with Progress() as progress:
                task = progress.add_task("[red]Running classical optimization...", total=config["restarts"])
                
                for _ in range(config["restarts"]):
                    params = np.random.randn(config["params"]).astype(np.float64) * 0.1
                    current_params = params.copy()
                    loss_fn = self.create_loss_landscape(complexity)
                    
                    for _ in range(config["iterations"]):
                        gradient = self.calculate_gradient(loss_fn, current_params)
                        current_params = current_params - 0.1 * gradient
                        current_loss = loss_fn(current_params)
                        best_classical_loss = min(best_classical_loss, current_loss)
                    
                    progress.update(task, advance=1)
            
            classical_time = time.time() - classical_start
            
            # Quantum-assisted optimization with timing breakdown
            qubo_start = time.time()
            loss_fn = self.create_loss_landscape(complexity)
            initial_params = np.random.randn(config["params"]).astype(np.float64) * 0.1
            gradient = self.calculate_gradient(loss_fn, initial_params)
            qubo = self.construct_qubo(gradient, initial_params)
            qubo_time = time.time() - qubo_start
            
            annealing_start = time.time()
            if dry_run:
                dry_run_compute += 1
                self.check_embedding()
            else:
                response = self.sampler.sample_qubo(qubo, num_reads=100)
            annealing_time = time.time() - annealing_start
            
            optimization_start = time.time()
            if not dry_run:
                param_history, loss_history = self.optimize(
                    num_params=config["params"],
                    max_steps=config["iterations"],
                    complexity=complexity
                )
            optimization_time = time.time() - optimization_start
            
            if not dry_run:
                quantum_loss = loss_history[-1]
                
                # Print detailed results
                self.console.print(f"\nProblem size: {config['params']} parameters")
                self.console.print(f"Classical optimization time: {classical_time:.3f}s")
                self.console.print(f"Best classical loss: {best_classical_loss:.6f}")
                self.console.print("\nQuantum timing breakdown:")
                self.console.print(f"  QUBO construction: {qubo_time:.3f}s")
                self.console.print(f"  Annealing time: {annealing_time:.3f}s")
                self.console.print(f"  Total optimization time: {optimization_time:.3f}s")
                self.console.print(f"Best quantum loss: {quantum_loss:.6f}")

                total_time = classical_time + optimization_time
                self._print_comparison(
                    best_classical_loss,
                    quantum_loss, 
                    classical_time,
                    total_time
                )
            if self.backend_type == "simulator":
                self.console.print("\n[dim]Note: Using simulator - real quantum hardware may show different timing patterns[/dim]")
        if dry_run:
            self.console.print(f"\n[cyan]Dry run complete: {dry_run_compute} QUBO problems checked for embedding")
            self.console.print("[dim]Note: Embedding check is only necessary for real quantum hardware")
            self.estimate_hardware_time(dry_run_compute, self.shots * dry_run_compute)

    def run_scaling_benchmark(self, max_param_size: int = 500, dry_run: bool = True, 
                            save_results: bool = True, adjust_iterations: bool = True):
        """
        Run comprehensive scaling benchmarks across different optimization approaches.
        
        Args:
            max_param_size: Maximum number of parameters to test (limited by hardware)
            dry_run: If True, use simulators instead of actual quantum hardware
            save_results: If True, save results to a JSON file for visualization
            adjust_iterations: If True, reduce iterations for larger parameter sizes
        """
        # Adjust iterations based on parameter size if requested
        original_iterations = self.config['iterations']
        if adjust_iterations:
            def get_adjusted_iterations(param_size):
                return max(10, int(original_iterations * (10 / param_size)))
                
        import json
        import os
        # Initialize cost tracking
        estimated_qpu_time = 0  # microseconds
        estimated_cost = 0.0  # USD
        # Parameter sizes to test - exponential growth
        param_sizes = [10, 20, 50, 100, 200, 500]
        param_sizes = [p for p in param_sizes if p <= max_param_size]
        
        benchmark_results = {
            'classical': {'times': [], 'losses': [], 'operations': []},
            'quantum': {'times': [], 'losses': [], 'physical_qubits': [], 'embedding_times': []},
            'hybrid': {'times': [], 'losses': [], 'quantum_portions': []}
        }
        
        results_data = []
        
        with Progress() as overall_progress:
            total_benchmarks = len(param_sizes)
            overall_task = overall_progress.add_task(
                "[cyan]Running all benchmarks...", 
                total=total_benchmarks
            )
            
            for param_size in param_sizes:
                if adjust_iterations:
                    self.config['iterations'] = get_adjusted_iterations(param_size)
                
                self.console.print(f"\n[cyan]Benchmarking with {param_size} parameters:")
                self.console.print(f"Using {self.config['iterations']} iterations")
            
            # Test if problem fits on hardware
            qubit_estimate = self._estimate_required_qubits(param_size)
            if qubit_estimate > 5000:  # D-Wave hardware limit
                self.console.print(f"[yellow]Warning: {param_size} parameters would require ~{qubit_estimate} qubits")
                self.console.print("[yellow]Skipping QPU-only benchmark for this size")
                can_run_quantum = False
            else:
                can_run_quantum = True
            
            # 1. Classical Benchmark
            classical_metrics = self._benchmark_classical(param_size)
            benchmark_results['classical']['times'].append(classical_metrics['time'])
            benchmark_results['classical']['losses'].append(classical_metrics['loss'])
            benchmark_results['classical']['operations'].append(classical_metrics['operations'])
            
            # 2. Quantum Benchmark (if feasible)
            if can_run_quantum:
                quantum_metrics = self._benchmark_quantum(param_size)
                benchmark_results['quantum']['times'].append(quantum_metrics['time'])
                benchmark_results['quantum']['losses'].append(quantum_metrics['loss'])
                benchmark_results['quantum']['physical_qubits'].append(quantum_metrics['physical_qubits'])
                benchmark_results['quantum']['embedding_times'].append(quantum_metrics['embedding_time'])
            
            # 3. Hybrid Benchmark
            hybrid_metrics = self._benchmark_hybrid(param_size)
            benchmark_results['hybrid']['times'].append(hybrid_metrics['time'])
            benchmark_results['hybrid']['losses'].append(hybrid_metrics['loss'])
            benchmark_results['hybrid']['quantum_portions'].append(hybrid_metrics['quantum_portion'])
            
            self._print_size_summary(param_size, classical_metrics, 
                                quantum_metrics if can_run_quantum else None,
                                hybrid_metrics)
            
            # Save results for this parameter size
            result_entry = {
                'paramSize': param_size,
                'classicalTime': classical_metrics['time'],
                'classicalLoss': classical_metrics['loss'],
                'classicalOps': classical_metrics['operations']
            }
            
            if can_run_quantum:
                result_entry.update({
                    'quantumTime': quantum_metrics['time'],
                    'quantumLoss': quantum_metrics['loss'],
                    'physicalQubits': quantum_metrics['physical_qubits'],
                    'embeddingTime': quantum_metrics['embedding_time'],
                    'estimatedQPUTime': quantum_metrics['estimated_qpu_time'],
                    'estimatedCost': quantum_metrics['estimated_cost']
                })
                
            result_entry.update({
                'hybridTime': hybrid_metrics['time'],
                'hybridLoss': hybrid_metrics['loss'],
                'hybridQuantumPortion': hybrid_metrics['quantum_portion']
            })
            
            results_data.append(result_entry)
            overall_progress.update(overall_task, advance=1)
            
        # Restore original iterations
        if adjust_iterations:
            self.config['iterations'] = original_iterations
            
        # Save results if requested
        if save_results:
            results_file = 'benchmark_results.json'
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2)
            self.console.print(f"\n[green]Results saved to {results_file}")
        
        return benchmark_results

    def _estimate_required_qubits(self, param_size: int) -> int:
        """
        Estimate physical qubits needed for a given parameter size.
        """
        # Each parameter needs log2(resolution) logical qubits
        bits_per_param = 8  # 8-bit resolution
        logical_qubits = param_size * bits_per_param
        
        # Estimate physical qubits needed based on typical embedding overhead
        # This is a rough estimate - actual requirements depend on problem structure
        embedding_factor = 3  # Typical factor for Pegasus architecture
        return logical_qubits * embedding_factor

    def _benchmark_classical(self, param_size: int) -> dict:
        """
        Run classical optimization benchmark.
        """
        start_time = time.time()
        operations = 0
        
        # Create test problem
        loss_fn = self.create_loss_landscape("complex")
        params = np.random.randn(param_size).astype(np.float64) * 0.1
        
        best_loss = float('inf')
        for _ in range(self.config['iterations']):
            gradient = self.calculate_gradient(loss_fn, params)
            params = params - 0.1 * gradient
            current_loss = loss_fn(params)
            best_loss = min(best_loss, current_loss)
            
            # Count operations (gradient computation + update)
            operations += param_size * 2  # Approximate operation count
        
        return {
            'time': time.time() - start_time,
            'loss': best_loss,
            'operations': operations
        }

    def _benchmark_quantum(self, param_size: int, dry_run: bool = True) -> dict:
        """
        Run quantum optimization benchmark.
        """
        start_time = time.time()
        embedding_start = time.time()
        
        # Construct initial QUBO
        loss_fn = self.create_loss_landscape("complex")
        params = np.random.randn(param_size).astype(np.float64) * 0.1
        gradient = self.calculate_gradient(loss_fn, params)
        qubo = self.construct_qubo(gradient, params)
        
        # Check embedding
        dw_sampler = DWaveSampler()
        embedding = minorminer.find_embedding(qubo.keys(), dw_sampler.edgelist)
        embedding_time = time.time() - embedding_start
        
        if not embedding:
            raise ValueError(f"Could not find embedding for {param_size} parameters")
        
        physical_qubits = sum(len(chain) for chain in embedding.values())
        
        # Calculate QPU time and cost estimates
        ANNEALING_TIME_US = 20  # microseconds
        READOUT_TIME_US = 100   # microseconds per shot
        PROGRAMMING_TIME_US = 10000  # microseconds per problem
        
        qpu_time_per_iteration = (
            PROGRAMMING_TIME_US +
            self.shots * (ANNEALING_TIME_US + READOUT_TIME_US)
        )
        total_qpu_time = qpu_time_per_iteration * self.config['iterations']
        
        # Cost estimation (based on LEAP pricing)
        BASE_COST_PER_MIN = 0.30
        SHOT_COST = 0.00145
        estimated_cost = (
            (total_qpu_time / 60e6) * BASE_COST_PER_MIN +  # Convert microseconds to minutes
            self.shots * self.config['iterations'] * SHOT_COST
        )
        
        # Run optimization
        best_loss = float('inf')
        for _ in range(self.config['iterations']):
            if dry_run:
                # Use simulated annealing sampler for dry run
                response = dimod.SimulatedAnnealingSampler().sample_qubo(
                    qubo, num_reads=self.shots
                )
            else:
                response = self.sampler.sample_qubo(qubo, num_reads=self.shots)
                
            best_sample = next(response.samples())
            params = self.decode_solution(best_sample, param_size)
            current_loss = loss_fn(params)
            best_loss = min(best_loss, current_loss)
        
        return {
            'time': time.time() - start_time,
            'loss': best_loss,
            'physical_qubits': physical_qubits,
            'embedding_time': embedding_time,
            'estimated_qpu_time': total_qpu_time,
            'estimated_cost': estimated_cost
        }

    def _benchmark_hybrid(self, param_size: int, dry_run: bool = True) -> dict:
        """
        Run hybrid optimization benchmark using D-Wave's hybrid solver.
        """
        start_time = time.time()
        
        # Initialize solver based on dry_run status
        if dry_run:
            hybrid_sampler = dimod.SimulatedAnnealingSampler()
            self.console.print("[yellow]Using simulated annealing for hybrid dry run")
        else:
            hybrid_sampler = LeapHybridSampler()
        
        # Create test problem
        loss_fn = self.create_loss_landscape("complex")
        params = np.random.randn(param_size).astype(np.float64) * 0.1
        
        best_loss = float('inf')
        quantum_time = 0
        
        for _ in range(self.config['iterations']):
            # Hybrid optimization step
            gradient = self.calculate_gradient(loss_fn, params)
            qubo = self.construct_qubo(gradient, params)
            
            # Track quantum portion of hybrid solve
            q_start = time.time()
            response = hybrid_sampler.sample_qubo(qubo)
            quantum_time += time.time() - q_start
            
            best_sample = next(response.samples())
            params = self.decode_solution(best_sample, param_size)
            current_loss = loss_fn(params)
            best_loss = min(best_loss, current_loss)
        
        total_time = time.time() - start_time
        
        return {
            'time': total_time,
            'loss': best_loss,
            'quantum_portion': quantum_time / total_time
        }

    def _print_size_summary(self, param_size: int, classical: dict, quantum: dict, hybrid: dict, dry_run: bool = True):
        """Print summary of benchmarks for a given parameter size."""
        console = Console()
        console.print(f"\n[bold]Results for {param_size} parameters:[/bold]")
        
        # Classical results
        console.print("\n[red]Classical:")
        console.print(f"  Time: {classical['time']:.2f}s")
        console.print(f"  Final loss: {classical['loss']:.6f}")
        console.print(f"  Operations: {classical['operations']:,}")
        
        # Quantum results (if available)
        if quantum:
            console.print("\n[blue]Quantum:")
            console.print(f"  Time: {quantum['time']:.2f}s")
            console.print(f"  Final loss: {quantum['loss']:.6f}")
            console.print(f"  Physical qubits: {quantum['physical_qubits']}")
            console.print(f"  Embedding time: {quantum['embedding_time']:.2f}s")
            if dry_run:
                console.print(f"  [dim]Estimated QPU time: {quantum['estimated_qpu_time']/1e6:.2f}s")
                console.print(f"  [dim]Estimated cost: ${quantum['estimated_cost']:.2f}")
        
        # Hybrid results
        console.print("\n[green]Hybrid:")
        console.print(f"  Time: {hybrid['time']:.2f}s")
        console.print(f"  Final loss: {hybrid['loss']:.6f}")
        console.print(f"  Quantum portion: {hybrid['quantum_portion']*100:.1f}%\n")

if __name__ == "__main__":
    optimizer = GradientWave()
    optimizer.run_demo()