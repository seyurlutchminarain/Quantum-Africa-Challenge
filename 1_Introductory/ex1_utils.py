import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

def estimate_number_of_qubits_required_for(
    max_hectares_per_crop=1,
    hectares_available=3,
):
    return 4 * np.ceil(np.log2(max_hectares_per_crop + 1)) + np.ceil(
        np.log2(hectares_available + 1)
    )
