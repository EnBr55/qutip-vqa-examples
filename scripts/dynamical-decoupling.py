import numpy as np
from qutip import Qobj
from qutip_qip.operations import z_gate, x_gate
from qutip_qip.vqa import VQA, VQA_Block

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

res = VQA_circuit.optimize_parameters(
        method='Powell',
        initial='random',
        bounds=[(0, np.inf) for _ in range(4*k)]
        )
print(res)
print('Total time', sum(res.angles))
