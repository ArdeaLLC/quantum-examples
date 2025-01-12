from quantum_examples.gradient_wave.gradient_wave import GradientWave
from quantum_examples.shor_lock_picker.quantum_lock import QuantumLockPicker
from quantum_examples.visualization import BenchmarkVisualizer

# 'dry_run', 'simulator', 'hybrid', 'hardware'
RUN_TYPE = 'dry_run'

def show_dwave_example():
    dwave_gradient_wave = GradientWave(backend_type=RUN_TYPE)
    dwave_gradient_wave.run_demo()

def show_shor_example():
    all_shored_up = QuantumLockPicker(backend_type=RUN_TYPE)
    all_shored_up.run_demo()


def main():
    # show_shor_example(FOR_REALZ)
    show_dwave_example()

if __name__ == '__main__':
    main()
