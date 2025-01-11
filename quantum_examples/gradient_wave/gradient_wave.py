from typing import List, Tuple, Optional
import numpy as np
import dimod
from dwave.system import DWaveSampler, EmbeddingComposite
import time
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress

class GradientWave:
    """
    Demonstrates using D-Wave quantum annealing for optimizing model parameters
    by mapping gradient descent to a QUBO problem.
    """
    
    def __init__(self, backend_type: str = "simulator"):
        """
        Initialize the Gradient Wave optimizer.
        
        Args:
            backend_type (str): Type of quantum backend to use ("simulator" or "hardware")
        """
        self.console = Console()
        self.backend_type = backend_type
        
        # Initialize D-Wave sampler
        if backend_type == "hardware":
            self.sampler = EmbeddingComposite(DWaveSampler())
            self.console.print("[yellow]Warning: Using real D-Wave quantum hardware!")
        else:
            self.sampler = dimod.SimulatedAnnealingSampler()
    
    def create_loss_landscape(self, complexity: str = "simple") -> callable:
        """
        Create a toy loss landscape function for optimization.
        
        Args:
            complexity (str): Complexity of the loss landscape ("simple", "medium", "complex")
            
        Returns:
            callable: Loss function that takes parameters and returns loss value
        """
        if complexity == "simple":
            # Simple quadratic bowl
            return lambda params: np.sum(params ** 2)
        elif complexity == "medium":
            # Multiple local minima
            return lambda params: np.sum(params ** 2) + 0.5 * np.sin(4 * np.pi * params[0])
        else:
            # Complex landscape with multiple local minima and barriers
            return lambda params: (np.sum(params ** 2) + 
                0.5 * np.sin(4 * np.pi * params[0]) + 
                0.3 * np.cos(3 * np.pi * params[1])
            )
    
    def calculate_gradient(self, loss_fn: callable, params: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
        """
        Calculate numerical gradient of the loss function.
        
        Args:
            loss_fn (callable): Loss function
            params (np.ndarray): Current parameters
            epsilon (float): Small value for numerical gradient calculation
            
        Returns:
            np.ndarray: Gradient vector
        """
        gradient = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += epsilon
            params_minus = params.copy()
            params_minus[i] -= epsilon
            gradient[i] = (loss_fn(params_plus) - loss_fn(params_minus)) / (2 * epsilon)
        return gradient
    
    def construct_qubo(
        self,
        gradient: np.ndarray,
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
        
        return qubo
    
    def decode_solution(self, sample: dict, num_params: int) -> np.ndarray:
        """
        Decode QUBO solution back to parameter space.
        
        Args:
            sample (dict): Solution from quantum annealer
            num_params (int): Number of parameters
            
        Returns:
            np.ndarray: Optimized parameters
        """
        return np.array([sample[f'p{i}'] for i in range(num_params)])
    
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
        params = np.random.randn(num_params)  # Random starting point
        param_history = [params.copy()]
        loss_history = [loss_fn(params)]
        
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
                best_sample = next(response.data(['sample', 'energy']).itertuples())
                new_params = self.decode_solution(best_sample.sample, num_params)
                
                # Update parameters
                params = new_params
                param_history.append(params.copy())
                loss_history.append(loss_fn(params))
                
                progress.update(task, advance=1)
                
                # Early stopping if converged
                if step > 0 and abs(loss_history[-1] - loss_history[-2]) < 1e-6:
                    break
        
        return param_history, loss_history
    
    def run_demo(self):
        """Run an interactive demo comparing classical and quantum optimization."""
        self.console.print(Panel.fit(
            "[bold cyan]Welcome to Gradient Wave![/bold cyan]\n"
            "Using D-Wave quantum annealing to optimize neural network parameters"
        ))
        
        # Run optimization with different complexities
        for complexity in ["simple", "medium", "complex"]:
            self.console.print(f"\n[cyan]Testing {complexity} loss landscape:")
            
            # Classical gradient descent
            classical_start = time.time()
            params = np.random.randn(2)
            loss_fn = self.create_loss_landscape(complexity)
            for _ in range(10):
                gradient = self.calculate_gradient(loss_fn, params)
                params = params - 0.1 * gradient
            classical_time = time.time() - classical_start
            
            # Quantum-assisted optimization
            quantum_start = time.time()
            param_history, loss_history = self.optimize(complexity=complexity)
            quantum_time = time.time() - quantum_start
            
            # Print results
            self.console.print(f"Classical optimization time: {classical_time:.3f}s")
            self.console.print(f"Quantum optimization time: {quantum_time:.3f}s")
            self.console.print(f"Final loss: {loss_history[-1]:.6f}")
            
            if complexity == "complex":
                self.console.print("\n[green]Notice how quantum annealing helps avoid local minima!")

if __name__ == "__main__":
    optimizer = GradientWave()
    optimizer.run_demo()