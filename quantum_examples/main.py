import os

from quantum_examples.gradient_wave.gradient_wave import GradientWave
from quantum_examples.shor_lock_picker.quantum_lock import QuantumLockPicker
from quantum_examples.visualization import BenchmarkVisualizer

os.environ['PYTHONUNBUFFERED'] = '1'
FOR_REALZ = False

def show_dwave_example(for_realz=False):
    if for_realz:
        dwave_gradient_wave = GradientWave(backend_type='hardware')
    else:
        dwave_gradient_wave = GradientWave()
    # dwave_gradient_wave.run_demo(dry_run=True)
    # dwave_gradient_wave.run_demo()
    results = dwave_gradient_wave.run_scaling_benchmark(max_param_size=50, dry_run=True, adjust_iterations=True)
    viz = BenchmarkVisualizer('benchmark_results.json')
    viz.generate_report()

def show_shor_example(for_realz=False):
    if for_realz:
        all_shored_up = QuantumLockPicker(backend_type='hardware')
    else:
        all_shored_up = QuantumLockPicker()
    all_shored_up.run_demo()


def main():
    # show_shor_example(FOR_REALZ)
    show_dwave_example(FOR_REALZ)

if __name__ == '__main__':
    main()
