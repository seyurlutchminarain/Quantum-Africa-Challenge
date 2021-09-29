from qiskit_nature.drivers import Molecule
from functools import partial as apply_variation_to_atom_pair


def lab3_ex1_valid():
    atom_pair = (6, 1)
    geometry = [
        ["O", [1.1280, 0.2091, 0.0000]],
        ["N", [-1.1878, 0.1791, 0.0000]],
        ["C", [0.0598, -0.3882, 0.0000]],
        ["H", [-1.3085, 1.1864, 0.0001]],
        ["H", [-2.0305, -0.3861, -0.0001]],
        ["H", [-0.0014, -1.4883, -0.0001]],
        ["C", [-0.1805, 1.3955, 0.0000]],
    ]
    charge = 0
    multiplicity = 1

    specific_molecular_variation = apply_variation_to_atom_pair(
        Molecule.absolute_stretching, atom_pair=atom_pair
    )
    macro_molecule = Molecule(
        geometry=geometry,
        charge=charge,
        multiplicity=multiplicity,
        degrees_of_freedom=[specific_molecular_variation],
    )
    return macro_molecule


def lab3_ex1_invalid():
    atom_pair = (5, 2)  # '(6,1)' -> '(5,2)'
    geometry = [
        ["O", [1.1280, 0.2091, 0.0000]],
        ["n", [-1.1878, 0.1791, 0.0000]],
        ["C", [0.0598, -0.3882, 0.0000]],
        ["H", [-1.3085, -1.1864, 0.0001]],
        ["H", [-2.0305, -0.3861, 0]],
        ["H", [-0.0014, -1.4883, -0.0001]],
        ["C", [-0.1805, 1.3955, 0.0000]],
    ]  # 'N' -> 'n', '1.1864' -> '-1.1864', '-0.0001' -> '0'
    charge = 1  # '0' -> '1'
    multiplicity = 2  # '1' -> '2'

    specific_molecular_variation = apply_variation_to_atom_pair(
        Molecule.absolute_stretching, atom_pair=atom_pair
    )
    macro_molecule = Molecule(
        geometry=geometry,
        charge=charge,
        multiplicity=multiplicity,
        degrees_of_freedom=[specific_molecular_variation],
    )
    return macro_molecule


def lab3_ex2_valid():
    return "above", "increase"


def lab3_ex2_invalid():
    return "below", "decrease"
