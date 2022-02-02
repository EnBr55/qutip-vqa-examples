import numpy as np
from qutip import Qobj, rand_herm, qeye
from qutip_qip.operations import z_gate, x_gate
from qutip_qip.vqa import VQA, VQA_Block
from scipy.optimize import LinearConstraint
from qutip.qip.operations.gates import gate_sequence_product

#H = rand_herm(N=2)
#H = H - H.tr()/2 * qeye(2)

# Traceless, Hermitian operator
H = Qobj([
    [1, 1],
    [1, -1]
    ])

k = 1

VQA_circuit = VQA(
            n_qubits=1,
            n_layers=4*k,
            cost_method="STATE",
        )

# One 'block' is given by (Z e^{-iH t/8k} X e^{-iH t/8k})^{4k}

def get_layer_unitary(t):
    U = (-1j * H * t/(8*k)).expm()
    return (z_gate() * U * x_gate() * U)


# overlap of initial state with final state
def state_overlap_cost(state):
    initial_state = VQA_circuit.get_initial_state()
    overlap_prob = abs(state.overlap(initial_state))**2
    return 1 - overlap_prob


VQA_circuit.cost_func = state_overlap_cost

U_block = VQA_Block(get_layer_unitary, is_unitary=True, name="U(t)")
VQA_circuit.add_block(U_block)

# Set lower and upper bounds to get the time as close to 1 as possible

def constraint_lb(t):
    return sum(t) - 0.999

def constraint_ub(t):
    return 1.001 - sum(t)

res = VQA_circuit.optimize_parameters(
        method='COBYLA',
        initial='random',
        constraints=[
            {
            'type': 'ineq',
            'fun': constraint_lb
            },
            {
            'type': 'ineq',
            'fun': constraint_ub
            },
            ],
        )
print(res)
print('Total time', sum(res.angles))
print(
    'Total unitary',
    gate_sequence_product(
        VQA_circuit.construct_circuit(res.res.x).propagators())
    )
