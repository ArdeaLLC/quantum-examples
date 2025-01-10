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
    def __init__(self, backend_type: str = "simulator"):
        """
        Initialize the Quantum Lock Picker.
        
        Args:
            backend_type (str): Type of quantum backend to use ("simulator" or "hardware")
        """
        self.console = Console()
        self.backend_type = backend_type
        self.device = self._initialize_device()
        
    def _initialize_device(self) -> AwsDevice:
        """Initialize the quantum device based on backend type."""
        if self.backend_type == "simulator":
            return AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")  # AWS SV1 simulator
        else:
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
        # For demo purposes, we'll implement a simplified version
        # In a real implementation, this would use quantum phase estimation
        
        # Create quantum circuit for period finding
        n_qubits = 2 * len(bin(N)[2:])  # Double the number of bits needed to represent N
        circuit = Circuit()
        
        # Add quantum operations (simplified for demo)
        for i in range(n_qubits // 2):
            circuit.h(i)  # Apply Hadamard gates to create superposition
        
        # Add measurement
        circuit.probability()
        
        # Execute circuit
        with Progress() as progress:
            task = progress.add_task("[cyan]Running quantum circuit...", total=100)
            
            try:
                result = self.device.run(circuit, shots=shots).result()
                progress.update(task, completed=100)
                
                # Process results (simplified)
                # In a real implementation, we would process the quantum measurements
                # to find the period
                
                # For demo, return a simplified period calculation
                return self._calculate_period_from_result(result, N)
                
            except Exception as e:
                self.console.print(f"[red]Error in quantum execution: {str(e)}")
                return None

    def _calculate_period_from_result(self, result, N: int) -> Optional[int]:
        """Calculate period from quantum measurement results."""
        # Simplified period calculation for demo
        # In a real implementation, this would process the quantum measurements
        # to find the actual period
        
        # For demo purposes, we'll return a value that would help factor N
        for r in range(2, N):
            if pow(2, r, N) == 1:
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
            
        # Find period using quantum circuit
        r = self.quantum_period_finding(N)
        if r is None:
            return None, None
            
        # Calculate factors
        if r % 2 == 0:
            candidate = pow(2, r//2, N)
            factor1 = math.gcd(candidate + 1, N)
            factor2 = math.gcd(candidate - 1, N)
            if factor1 > 1 and factor2 > 1:
                return factor1, factor2
                
        return None, None

    def run_demo(self):
        """Run an interactive demo of the quantum lock picker."""
        self.console.print(Panel.fit(
            "[bold cyan]Welcome to Sin's Quantum Lock Picker![/bold cyan]\n"
            "Using Shor's Algorithm to break digital locks quantum-style!"
        ))
        
        # Create a new lock
        p1, p2, lock_number = self.create_lock()
        self.console.print(f"\n[green]Created a new quantum lock with number: {lock_number}")
        
        # Classical attempt visualization
        self.console.print("\n[yellow]First, let's try breaking it classically...")
        with Progress() as progress:
            task = progress.add_task("[red]Trying classical factoring...", total=100)
            for i in range(100):
                time.sleep(0.05)  # Simulate classical computation time
                progress.update(task, advance=1)
                
        # Now quantum attempt
        self.console.print("\n[cyan]Now, let's use quantum computing with Shor's Algorithm!")
        factors = self.factor_number(lock_number)
        
        if factors[0] is not None:
            self.console.print(f"\n[bold green]Success! Lock broken!")
            self.console.print(f"Factors found: {factors[0]} × {factors[1]} = {lock_number}")
            if factors[0] == p1 and factors[1] == p2 or factors[0] == p2 and factors[1] == p1:
                self.console.print("[bold green]✓ Correct factors found!")
            else:
                self.console.print("[bold red]× Different factorization found!")
        else:
            self.console.print("[bold red]Could not break the lock this time.")
    
    def test_class(self):
        print("Class Loaded")

if __name__ == "__main__":
    # Create and run the demo
    lock_picker = QuantumLockPicker()
    lock_picker.test_class()
    # lock_picker.run_demo()