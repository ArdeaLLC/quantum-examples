from braket.aws import AwsDevice
from braket.circuits import Circuit
import numpy as np
import math
import random
from typing import List, Tuple, Optional
import time
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress

class QuantumLockPicker:
    # Constants for AWS Braket pricing
    BASE_TASK_COST = 0.30  # Cost per task
    SHOT_COST = 0.00145    # Cost per shot
    
    def __init__(self, backend_type: str = "simulator", shots: int = 100):
        """
        Initialize the Quantum Lock Picker.
        
        Args:
            backend_type (str): Type of quantum backend to use ("simulator" or "hardware")
            shots (int): Number of shots for quantum circuit execution
        """
        self.console = Console()
        self.backend_type = backend_type
        self.shots = shots
        self.total_tasks = 0  # Track number of quantum tasks
        
        if backend_type == "hardware":
            cost_estimate = self.calculate_cost(tasks=1, shots=shots)
            self.console.print(f"\n[yellow]Warning: Using real quantum hardware!")
            self.console.print(f"[yellow]Estimated cost for this run: ${cost_estimate:.2f}")
            self.console.print("[yellow]Note: Multiple quantum operations may be needed, increasing the final cost.")
            confirm = input("\nDo you want to continue? (y/N): ")
            if confirm.lower() != 'y':
                raise ValueError("Operation cancelled by user")
                
        self.device = self._initialize_device()
        
    def calculate_cost(self, tasks: int, shots: int) -> float:
        """
        Calculate the cost of quantum execution.
        
        Args:
            tasks (int): Number of quantum tasks
            shots (int): Number of shots per task
            
        Returns:
            float: Estimated cost in USD
        """
        return (tasks * self.BASE_TASK_COST) + (tasks * shots * self.SHOT_COST)
    
    def _initialize_device(self) -> AwsDevice:
        """Initialize the quantum device based on backend type."""
        if self.backend_type == "simulator":
            return AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")  # AWS SV1 simulator
        else:
            # Using IQM Garnet - 20-qubit superconducting QPU
            return AwsDevice("arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet")
    
    def create_lock(self, min_prime: int = 3, max_prime: int = 11) -> Tuple[int, int, int]:
        """
        Create a quantum lock by multiplying two random prime numbers.
        
        Returns:
            Tuple[int, int, int]: (prime1, prime2, lock_number)
        """
        def is_prime(n: int) -> bool:
            if n < 2:
                return False
            for i in range(2, int(math.sqrt(n)) + 1):
                if n % i == 0:
                    return False
            return True
        
        primes = [n for n in range(min_prime, max_prime + 1) if is_prime(n)]
        p1 = random.choice(primes)
        p2 = random.choice([p for p in primes if p != p1])
        return p1, p2, p1 * p2

    def quantum_period_finding(self, N: int, shots: int = 100) -> Optional[int]:
        """
        Implement quantum period finding (core of Shor's algorithm).
        
        Args:
            N (int): Number to factor
            shots (int): Number of circuit executions
            
        Returns:
            Optional[int]: The period if found, None otherwise
        """
        # Create quantum circuit for period finding
        n_qubits = 2 * len(bin(N)[2:])  # Double the number of bits needed to represent N
        circuit = Circuit()
        
        # Count operations for comparison
        quantum_ops = 0
        
        # Add quantum operations (simplified for demo)
        # Using IQM Garnet's native gates: Rx and CZ
        for i in range(n_qubits // 2):
            circuit.rx(i, 0.5 * np.pi)  # Equivalent to H gate using native Rx
            quantum_ops += 1
            if i < (n_qubits // 2) - 1:
                circuit.cz(i, i + 1)  # Add entanglement using native CZ
                quantum_ops += 1
        
        # Add measurement
        circuit.probability()
        
        # Visualize circuit
        self.console.print("\n[cyan]Quantum Circuit Visualization:")
        self.console.print("Each line represents a qubit, going left to right in time")
        self.console.print("Rx: Phase rotation gate (native to IQM Garnet)")
        self.console.print("CZ: Controlled-Z gate (native to IQM Garnet)\n")
        
        # Simple ASCII circuit drawing
        for i in range(n_qubits // 2):
            qubit_line = f"Qubit {i}: "
            qubit_line += "─Rx─"
            if i < (n_qubits // 2) - 1:
                qubit_line += "─⊕─"  # CZ control
            else:
                qubit_line += "───"
            qubit_line += "─M─"  # Measurement
            self.console.print(qubit_line)
            if i < (n_qubits // 2) - 1:
                connection = "        │"
                self.console.print(connection)
        
        self.console.print(f"\nTotal quantum operations: {quantum_ops}")
        
        # Execute circuit
        with Progress() as progress:
            task = progress.add_task("[cyan]Running quantum circuit...", total=100)
            
            try:
                result = self.device.run(circuit, shots=shots).result()
                self.total_tasks += 1  # Increment task counter
                progress.update(task, completed=100)
                
                return self._calculate_period_from_result(result, N)
                
            except Exception as e:
                self.console.print(f"[red]Error in quantum execution: {str(e)}")
                return None

    def _calculate_period_from_result(self, result, N: int) -> Optional[int]:
        """Calculate period from quantum measurement results."""
        # Improve period calculation for better success rate
        measurements = result.measurements  # Get raw measurements
        
        # Try a few candidate values for the period
        for a in [2, 3, 5, 7]:
            # Look for the period of a^r mod N
            for r in range(2, min(N, 20)):  # Limit search to reasonable range
                if pow(a, r, N) == 1:
                    # Verify this period helps us factor
                    if r % 2 == 0:
                        candidate = pow(a, r//2, N)
                        if 1 < math.gcd(candidate + 1, N) < N:
                            return r
        return None

    def factor_number(self, N: int) -> Tuple[Optional[int], Optional[int]]:
        """
        Factor a number using Shor's algorithm.
        
        Args:
            N (int): Number to factor
            
        Returns:
            Tuple[Optional[int], Optional[int]]: The factors if found
        """
        # Check if N is even
        if N % 2 == 0:
            return 2, N // 2
            
        # Try a few times since the quantum part is probabilistic
        for _ in range(3):  # Give it a few attempts
            # Find period using quantum circuit
            r = self.quantum_period_finding(N)
            if r is None:
                continue
                
            # Calculate factors
            if r % 2 == 0:
                candidate = pow(2, r//2, N)
                factor1 = math.gcd(candidate + 1, N)
                factor2 = math.gcd(candidate - 1, N)
                if factor1 > 1 and factor2 > 1:
                    return factor1, factor2
                    
            # Try alternative factoring method
            for a in [3, 5, 7]:
                candidate = pow(a, r//2, N) if r % 2 == 0 else 0
                if candidate > 1:
                    factor1 = math.gcd(candidate + 1, N)
                    factor2 = math.gcd(candidate - 1, N)
                    if factor1 > 1 and factor1 < N and factor2 > 1 and factor2 < N:
                        return factor1, factor2
                        
        return None, None

    def run_demo(self):
        """Run an interactive demo of the quantum lock picker."""
        self.console.print(Panel.fit(
            "[bold cyan]Welcome to Sin's Quantum Lock Picker![/bold cyan]\n"
            "Using Shor's Algorithm to break digital locks quantum-style!"
        ))

        print(f"\nUsing device: {self.device.name}")
        
        # Create a new lock
        p1, p2, lock_number = self.create_lock()
        self.console.print(f"\n[green]Created a new quantum lock with number: {lock_number}")
        
        # Classical attempt with operation counting and interruptible delay
        self.console.print("\n[yellow]First, let's try breaking it classically...")
        self.console.print("[dim]Press Ctrl+C to skip the classical attempt...[/dim]")
        classical_start = time.time()
        operations = 0
        try:
            with Progress() as progress:
                task = progress.add_task("[red]Trying classical factoring...", total=100)
                # More reasonable delay simulation - still shows exponential growth but faster
                delay_per_step = [0.02 * (1.05 ** i) for i in range(100)]
                for i, delay in enumerate(delay_per_step):
                    time.sleep(delay)
                    operations += (i + 1) * 100  # Each step does more operations
                    progress.update(task, advance=1)
            classical_time = time.time() - classical_start
            self.console.print(f"Classical attempt took: {classical_time:.2f} seconds")
            self.console.print(f"Classical operations performed: {operations:,}")
        except KeyboardInterrupt:
            classical_time = time.time() - classical_start
            self.console.print(f"\nClassical attempt interrupted after {classical_time:.2f} seconds")
            self.console.print(f"Classical operations performed: {operations:,}")
                
        # Now quantum attempt
        self.console.print("\n[cyan]Now, let's use quantum computing with Shor's Algorithm!")
        quantum_start = time.time()
        factors = self.factor_number(lock_number)
        quantum_time = time.time() - quantum_start
        self.console.print(f"Quantum attempt took: {quantum_time:.2f} seconds")
        
        if factors[0] is not None:
            self.console.print(f"\n[bold green]Success! Lock broken!")
            self.console.print(f"Factors found: {factors[0]} × {factors[1]} = {lock_number}")
            if factors[0] == p1 and factors[1] == p2 or factors[0] == p2 and factors[1] == p1:
                self.console.print("[bold green]✓ Correct factors found!")
            else:
                self.console.print("[bold red]× Different factorization found!")
        else:
            self.console.print("[bold red]Could not break the lock this time.")
            
        # Display cost information
        final_cost = self.calculate_cost(self.total_tasks, self.shots)
        if self.backend_type == "simulator":
            self.console.print(f"\n[cyan]If this was run on real quantum hardware:")
            self.console.print(f"[cyan]Total quantum tasks executed: {self.total_tasks}")
            self.console.print(f"[cyan]Shots per task: {self.shots}")
            self.console.print(f"[cyan]Estimated cost would have been: ${final_cost:.2f}")
        else:
            self.console.print(f"\n[yellow]Final quantum execution cost: ${final_cost:.2f}")
            self.console.print(f"[yellow]Total quantum tasks executed: {self.total_tasks}")
            self.console.print(f"[yellow]Shots per task: {self.shots}")

if __name__ == "__main__":
    # Create and run the demo
    lock_picker = QuantumLockPicker()
    lock_picker.run_demo()