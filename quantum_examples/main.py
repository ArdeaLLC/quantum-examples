# import dimod
# from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
# from dwave.embedding import verify_embedding

from quantum_examples.shor_lock_picker.quantum_lock import QuantumLockPicker


def main():
    all_shored_up = QuantumLockPicker()
    # all_shored_up.test_class()
    all_shored_up.run_demo()

if __name__ == '__main__':
    main()
