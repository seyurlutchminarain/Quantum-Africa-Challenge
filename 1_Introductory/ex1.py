from qiskit_optimization import QuadraticProgram
from qiskit_optimization.problems.linear_constraint import LinearConstraint
from qiskit_optimization.problems.variable import Variable, VarType
import numpy as np
import logging as log


class QPArrayReorderer:
    def __init__(self, expected_variable_order: list, given_variable_order: list):
        self._expected = np.array(expected_variable_order)
        self._given = np.array(given_variable_order)
        self._index = [np.where(self._given == x)[0][0] for x in self._expected]

    def map(self, mat: np.ndarray):
        _mat = np.array(mat.copy())
        # If a 1D array, reorder columns
        if len(_mat.shape) == 1:
            _mat = _mat[self._index]
        elif len(_mat.shape) == 2:
            _mat = _mat[self._index, :][:, self._index]
            # Transpose check
            if not np.allclose(_mat, np.triu(_mat)):
                _mat = _mat.transpose()
            if not np.allclose(_mat, np.triu(_mat)):
                raise RuntimeError(
                    f"Reordering did not result in an upper triangular matrix: {_mat}"
                )
        else:
            raise NotImplementedError(
                "QPArrayReorderer cannot reorder arrays that aren't 1D or 2D"
            )

        return _mat


def expectedsolution() -> QuadraticProgram:
    ### Define crop-yield quadratic coefficients
    # Constraint variables on hectares
    max_hectares_per_crop = 1
    hectares_available = 3

    # Crops to be planted
    crops = ["Wheat", "Soybeans", "Maize", "PushPull"]

    # Monoculture crop yields, in tons/hectare
    yield_monoculture_per_hectare = {
        "Wheat": 2,
        "Soybeans": 1,
        "Maize": 4,
        "PushPull": 0,
    }

    # Intercrop crop yields, in tons/hectare
    yield_intercrop_per_hectare = {
        ("Wheat", "Soybeans"): 2.4,
        ("Wheat", "Maize"): 4,
        ("Wheat", "PushPull"): 4,
        ("Soybeans", "Maize"): 2,
        ("Soybeans", "PushPull"): 1,
        ("Maize", "PushPull"): 5,
    }
    # Create a QuadraticProgram
    cropyield = QuadraticProgram(
        name="Crop Yield",
    )

    # Add crop-yield variables
    for crop in crops:
        # Note that we add these as integer variables but max_hectares_per_crop is 1
        cropyield.integer_var(lowerbound=0, upperbound=max_hectares_per_crop, name=crop)

    # Add crop-yield quadratic using monoculture and intercrop yield variables
    cropyield.maximize(
        linear=yield_monoculture_per_hectare,
        quadratic=yield_intercrop_per_hectare,
    )

    # Add constraint for the total farm area
    hectares_available_linear_dict = dict([(crop, 1) for crop in crops])
    cropyield.linear_constraint(
        linear=hectares_available_linear_dict,
        sense="<=",
        rhs=hectares_available,
        name="Hectares Availabe",
    )
    return cropyield


def variablesdiffer(var1: Variable, var2: Variable):
    def isbinaryintegerpair(var_a, var_b):
        return any(
            v == VarType.INTEGER for v in [var_a.vartype, var_b.vartype]
        ) and any(v == VarType.BINARY for v in [var_a.vartype, var_b.vartype])

    _result = False
    if var1.lowerbound != var2.lowerbound:
        _result = True
    if var1.upperbound != var2.upperbound:
        _result = True
    if var1.vartype != var2.vartype and not isbinaryintegerpair(var1, var2):
        _result = True
    if isbinaryintegerpair(var1, var2):
        varbinary = var1 if var1.vartype == VarType.BINARY else var2
        varinteger = var1 if var1.vartype == VarType.INTEGER else var2
        # Check if integers bounds are not [0,1], i.e. a binary integer variable
        if varinteger.lowerbound != 0:
            _result = True
        if varinteger.upperbound != 1:
            _result = True
    if var1.name != var2.name:
        _result = True
    return _result


def anyvariablediffers(qp1, qp2):
    if len(qp1.variables) != len(qp2.variables):
        log.debug("Number of variables is different")
        return True
    for x in qp1.variables:
        var1 = qp1.get_variable(x.name)
        var2 = qp2.get_variable(x.name)
        if variablesdiffer(var1, var2):
            log.debug(f"variables qp1.{var1.name} and qp2.{var2.name} differ")
            return True
    return False


def objectivesdiffer(reord, obj1, obj2):
    if not np.all(obj1.linear.to_array() == reord.map(obj2.linear.to_array())):
        log.debug(
            f"Objective linear is incorrect:\n{obj1.linear.to_array()}\n{reord.map(obj2.linear.to_array())}"
        )
        return True
    if not np.all(obj1.quadratic.to_array() == reord.map(obj2.quadratic.to_array())):
        log.debug(
            f"Objective quadratic is incorrect:\n{obj1.quadratic.to_array()}\n{reord.map(obj2.quadratic.to_array())}"
        )
        return True
    if obj1.constant != obj2.constant:
        log.debug(f"Objective constant is incorrect:\n{obj1.constant}\n{obj2.constant}")
        return True
    if obj1.sense != obj2.sense:
        log.debug(f"Objective sense is incorrect:\n{obj1.sense}\n{obj2.sense}")
        return True
    return False


def ndarrayToTuple(mat: np.ndarray):
    if len(mat.shape) == 1:
        return tuple(list(mat))
    else:
        return tuple(list(mat.flatten()))


def numlinearconstraintsdiffer(qp1: QuadraticProgram, qp2: QuadraticProgram):
    return len(qp1.linear_constraints) != len(qp2.linear_constraints)


def numquadraticconstraintsdiffer(qp1: QuadraticProgram, qp2: QuadraticProgram):
    return len(qp1.quadratic_constraints) != len(qp2.quadratic_constraints)


def linearconstraintsdiffer(
    reord: QPArrayReorderer, qp1: QuadraticProgram, qp2: QuadraticProgram
):
    # Check quadratic constraints
    qp1_linear_constraints = set(
        [
            (c.sense, ndarrayToTuple(c.linear.to_array()), c.rhs)
            for c in qp1.linear_constraints
        ]
    )
    qp2_linear_constraints = set(
        [
            (c.sense, ndarrayToTuple(reord.map(c.linear.to_array())), c.rhs)
            for c in qp2.linear_constraints
        ]
    )
    _ans = qp1_linear_constraints != qp2_linear_constraints
    if _ans:
        log.debug(f"{qp1_linear_constraints}, {qp2_linear_constraints}")
    return _ans


def quadraticconstraintsdiffer(
    reord: QPArrayReorderer, qp1: QuadraticProgram, qp2: QuadraticProgram
):
    # Check quadratic constraints
    qp1_quadratic_constraints = set(
        [
            (
                c.sense,
                ndarrayToTuple(c.quadratic.to_array()),
                ndarrayToTuple(c.linear.to_array()),
                c.rhs,
            )
            for c in qp1.quadratic_constraints
        ]
    )
    qp2_quadratic_constraints = set(
        [
            (
                c.sense,
                ndarrayToTuple(reord.map(c.quadratic.to_array())),
                ndarrayToTuple(reord.map(c.linear.to_array())),
                c.rhs,
            )
            for c in qp2.quadratic_constraints
        ]
    )
    _ans = qp1_quadratic_constraints != qp2_quadratic_constraints
    if _ans:
        log.debug(f"{qp1_quadratic_constraints}, {qp2_quadratic_constraints}")
    return _ans


# Grading code
def check_ex1b(quadprog: QuadraticProgram):
    from qiskit_optimization.problems.variable import VarType
    from qiskit_optimization.problems.constraint import ConstraintSense as Sense

    _expected = expectedsolution()

    # Check variables
    if anyvariablediffers(_expected, quadprog):
        return False, "Variables are incorrect"

    _expected_vars = [v.name for v in _expected.variables]
    _given_vars = [v.name for v in quadprog.variables]
    reord = QPArrayReorderer(_expected_vars, _given_vars)

    # Check objective functions
    obj1 = _expected.objective
    obj2 = quadprog.objective
    if objectivesdiffer(reord, obj1, obj2):
        return False, "Objective function is incorrect"

    if numlinearconstraintsdiffer(_expected, quadprog):
        return False, "Incorrect number of linear constraints"

    if linearconstraintsdiffer(reord, _expected, quadprog):
        return False, "Linear constraints are incorrect"

    if numquadraticconstraintsdiffer(_expected, quadprog):
        return False, "Incorrect number of quadratic constraints"

    if quadraticconstraintsdiffer(reord, _expected, quadprog):
        return False, "Quadratic constraints are incorrect"

    # Everything is fine!
    return True, None


def grade_ex1b(quadraticProgram):
    correct, message = check_ex1b(quadraticProgram)
    if not correct:
        print(f"😢 Your answer is incorrect.")
        print(f"Message from grader: {message}")
        # return False

    if correct:
        print(f"🌟 Correct! Your quadratic program is a valid solution.")
        # return True


def check_ex1a(modulenames: list):
    _correct_names = set(
        [
            x.lower()
            for x in [
                "Qiskit Nature",
                "Qiskit Optimization",
                "Qiskit Finance",
                "Qiskit Machine Learning",
            ]
        ]
    )
    _given_names = set([x.lower() for x in modulenames])

    if len(_given_names) != len(_correct_names):
        return False, "Incorrect number of module names"

    if _correct_names != _given_names:
        return False, "Incorrect module names"

    return True, None


def grade_ex1a(modulenames: list):
    correct, message = check_ex1a(modulenames)
    if not correct:
        print(f"😢 Your answer is incorrect.")
        print(f"Message from grader: {message}")

    if correct:
        print(f"🌟 Correct! You found the correct module names!")


def check_ex1c(tonnage_qaoa: int, tonnage_vqe: int):
    if tonnage_vqe != 19.0 or tonnage_qaoa != 19.0:
        return False, "One or more of your answers is incorrect."
    return True, None


def grade_ex1c(tonnage_qaoa: int, tonnage_vqe: int):
    correct, message = check_ex1c(tonnage_qaoa, tonnage_vqe)
    if not correct:
        print(f"😢 Your answer is incorrect.")
        print(f"Message from grader: {message}")

    if correct:
        print(
            f"🌟 Correct! Your quantum algorithms calculated the correct maximum yield for the farm!"
        )
