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

if __name__ == "__main__":
    optimizer = GradientWave()
    optimizer.run_demo()