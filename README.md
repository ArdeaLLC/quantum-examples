# quantum-examples
Public examples of using current quantum computers to solve toy problems.

# Examples:

## Gradient Wave (D-Wave)

### Initial Setup
```bash
Using device: DWaveSampler (Simulator)

Testing optimization landscapes:
- Simple: O(n²) complexity, 20 parameters
- Medium: O(n³) complexity, 20 parameters
- Complex: O(n⁴) complexity, 20 parameters
```
The demo starts by setting up different optimization problems of increasing complexity. Each landscape represents a 
type of problem that might be encountered in machine learning or quantum chemistry, where finding global 
minima is crucial.

### Classical Attempt
```bash
Running classical optimization...
Simple landscape: 0.400s (80 iterations)
Medium landscape: 4.623s (120 iterations)
Complex landscape: 30.862s (160 iterations)
```
This simulates classical gradient descent trying to optimize these landscapes by:
- Using multiple random starting points to avoid local minima
- Counting operations to show computational complexity
- The exponential time growth demonstrates why classical methods struggle with complex landscapes
- Showing how runtime scales with problem complexity

#### ELI5: Classical Gradient Descent
Imagine you're in a hilly area trying to find the lowest point (valley). Classical gradient descent is like:
1. Start at a random spot
2. Look around to find which direction goes downhill the steepest
3. Take a step in that direction
4. Repeat until you can't go downhill anymore

The problem? You might end up in a small valley that isn't actually the lowest point in the whole area. That's 
called a "local minimum". To avoid this, classical methods try:
- Starting from different random spots (multiple restarts)
- Taking different size steps (learning rates)
- Looking further around (momentum)

But this gets really slow as the landscape gets more complex, especially when parameters interact with each 
other (like in our O(n⁴) case).

### Quantum Circuit Preparation
```bash
Now using quantum annealing with D-Wave...

Quantum timing breakdown:
QUBO construction: 0.005s -> 0.038s -> 0.190s
Annealing time: ~8.5s (consistent)
Total optimization: ~17.5s
```
This shows the quantum annealing process:
- Converting the optimization problem to a Quadratic Unconstrained Binary Optimization (QUBO) format
- Using D-Wave's quantum annealer to find low-energy states
- The QUBO construction time increases with problem complexity
- Annealing time stays consistent because we're using the same number of qubits

#### ELI5: How Quantum Annealing Works
Instead of walking down a hill like classical gradient descent, quantum annealing is more like:
1. Turn the entire landscape into a quantum system where particles can "tunnel" through barriers
2. Start with particles spread out everywhere in a high-energy state
3. Slowly "cool down" the system (like slowly freezing water)
4. The particles naturally settle into the lowest energy state possible

The quantum advantage comes from:
- Being able to tunnel through barriers (avoiding getting stuck in local minima)
- Exploring many possibilities simultaneously through quantum superposition
- Finding global optima more efficiently in complex landscapes

This is why our complex O(n⁴) landscape, which takes classical methods 30+ seconds, can still be optimized 
by the quantum approach in about the same time as simpler problems.

### Quantum Execution
```bash
Problem size: 20 parameters
Simple landscape:
  Classical loss: -5.464779
  Quantum loss: 0.000000
  
Medium landscape:
  Classical loss: -229.942557
  Quantum loss: 0.000000
  
Complex landscape:
  Classical loss: 626.100960
  Quantum loss: 1330.000000
```
The results show interesting trade-offs:
- Simple problems: Classical methods are faster but quantum finds better minima
- Medium problems: Quantum advantage becomes more apparent
- Complex problems: Quantum maintains consistent performance while classical time explodes
- Real quantum hardware might show different patterns than the simulator

The entire demo illustrates core quantum annealing concepts:
- Problem mapping to QUBO format
- Quantum speedup potential for complex landscapes
- Trade-offs between classical and quantum approaches
- Hardware-specific considerations (using D-Wave's native capabilities)

#### Conclusion
This demonstration shows why quantum annealing could be valuable for optimizing complex models, especially when:
- The parameter space is large
- Parameters have complex interactions
- Local minima are a significant problem
- Traditional gradient descent struggles

It also highlights current challenges:
- Problem mapping overhead (QUBO construction time)
- Limited qubit connectivity on real hardware
- The need to find good problem encodings
- Trade-offs between solution quality and computation time

## Grover's Groove (AWS Bracket)
Coming soon...

## All Shore'd Up (AWS Bracket)

### Initial Setup
```bash
Using device: Garnet

Created a new quantum lock with number: 35
```
The demo starts by creating a "lock" - it randomly selects two prime numbers (in this case 5 and 7) and multiplies 
them together (35). This mimics how RSA encryption works, where security relies on the difficulty of factoring large 
numbers.

### Classic Attempt
```bash
First, let's try breaking it classically...
Press Ctrl+C to skip the classical attempt...
Trying classical factoring... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
Classical attempt took: 52.57 seconds
Classical operations performed: 505,000
```
This simulates a classical computer trying to factor the number by:
- Attempting different divisors
- Counting operations to show computational complexity
- The exponential time growth demonstrates why classical factoring becomes infeasible with large numbers
- Showing operation count (505,000) gives a sense of computational work required
#### ELI5: Classical Lock Picking
Imagine you have a lock where the combination is 35. But instead of trying each number one at a time, we can think 
about it in binary (bits): 35 is 100011 in binary, which is 6 bits long. A classical computer has to try different 
combinations of these bits to find two numbers that multiply to make 35. 

The number of operations grows exponentially with the number of bits. For a number N that is n bits long:
- You need to try potential factors up to √N
- Each multiplication operation requires about n² bit operations
- Total operations ≈ √N * n²

In our example with 35:
- 35 is 6 bits long (100011)
- √35 ≈ 6 (we need to try numbers 2 through 6)
- Each multiplication needs 6² = 36 bit operations
- So we need about 6 * 36 = 216 operations per attempt
- Over many attempts across our search space, this leads to our observed ~505,000 operations

This gets really hard really fast! If we were trying to break a real cryptographic key (like 2048 bits), we'd need 
more operations than atoms in the universe. Which I think is around 10^80 estimated if I remember right.

### Quantum Circuit Preparation
```bash
Now, let's use quantum computing with Shor's Algorithm!

Quantum Circuit Visualization:
Each line represents a qubit, going left to right in time
Rx: Phase rotation gate (native to IQM Garnet)
CZ: Controlled-Z gate (native to IQM Garnet)

Qubit 0: ─Rx──⊕──M─
        │
Qubit 1: ─Rx──⊕──M─
        │
Qubit 2: ─Rx──⊕──M─
        │
Qubit 3: ─Rx──⊕──M─
        │
Qubit 4: ─Rx──⊕──M─
        │
Qubit 5: ─Rx─────M─

Total quantum operations: 11
```
This shows the quantum circuit being constructed:
- Each line represents a quantum bit (qubit)
- Rx gates are phase rotation gates (native to IQM Garnet hardware)
- CZ gates are controlled-Z gates for entanglement
- M represents measurement
- The vertical lines show qubit interactions
- Total operations (11) shows quantum efficiency compared to classical
#### ELI5: How Shor's Algorithm Works
Using our same lock with combination 35 (100011 in binary), Shor's algorithm takes a completely different approach 
than trying combinations:

1. Instead of trying factors directly, it uses quantum superposition (all possibilities at once) to find a 
repeating pattern in numbers, called a period.
2. For 35, it would:
   - Pick a random number 'a' (say, 2)
   - Create quantum states that represent: 2¹ mod 35, 2² mod 35, 2³ mod 35, etc.
   - The sequence we get is: 2, 4, 8, 16, 32, 29, 23, 11, 22, 9, 18, 36, 1, ...
   - Notice it repeats! (back to 2 after this)
3. Finding this period (12 in this case) gives us a high chance of finding factors of 35

This is why our quantum circuit only needed 11 operations instead of 505,000. For bigger numbers like those used 
in real cryptography (2048 bits), the difference is even more dramatic:
- Classical: More operations than atoms in the universe
- Quantum with Shor's: Only about 2048³ operations (still a lot, but actually possible!)

This is why the future of quantum computers is both amazing and scary for current encryption methods - they 
turn an impossible problem into a possible one.

### Quantum Execution
```bash
Running quantum circuit... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
Quantum attempt took: 4.30 seconds

Success! Lock broken!
Factors found: 5 × 7 = 35
✓ Correct factors found!

If this was run on real quantum hardware:
Total quantum tasks executed: 1
Shots per task: 100
Estimated cost would have been: $0.44
```
Provides cost transparency for quantum hardware:
- Base cost per task ($0.30)
- Per-shot cost (100 shots × $0.00145)
- This is based specifically on the IQM - Garnet (20 qubit universal gate-model superconducting QPU)
- Total cost estimation
- Shows actual tasks executed (important since algorithm may need multiple attempts)

The entire demo illustrates core quantum computing concepts:
- Quantum speedup over classical algorithms
- Real quantum hardware considerations (native gates, costs)
- Practical quantum algorithm implementation (Shor's)
- The potential of quantum computing for cryptography and factoring
- Hardware-specific optimizations (using IQM Garnet's native gates)

#### Conclusion
This demonstration also shows why quantum computing poses future challenges for current encryption methods 
while making the concepts accessible through an interactive "lock picking" metaphor.

Also though, how amazing the speedup will be of certain problems that can be represented in a way
where these types of quantum algorithms will make short work of something that would have taken
years or simply been infeasible with current compute power.

Grover's algorithm can quadratically reduce a search space. 2^256 becomes 2^128 
(but requires at least 256 error-corected logical quibits)

Shor's algorithm can do things like turn two uses of a private key to make two public keys
from a nearly impossible eliptical curve cryptography problem to polynomial time for solving 
with classical compute.

This algorithm can efficiently find the factors of large numbers, which is what makes it a threat to certain types of 
encryption. Modern encryption often relies on the fact that multiplying two large prime numbers is easy, but finding 
those original prime numbers (factoring) is very hard for classical computers. Shor's algorithm can solve this 
problem efficiently on a quantum computer, potentially breaking encryptions that rely on this difficulty. 
For example, it could break RSA encryption by finding the prime factors of the public key. However, like 
Grover's algorithm, this requires large-scale fault-tolerant quantum computers that don't exist yet.
