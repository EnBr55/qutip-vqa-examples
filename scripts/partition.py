import numpy as np
from qutip import Qobj, tensor
from qutip_qip.operations import z_gate, x_gate, snot
from qutip_qip.vqa import VQA, VQA_Block

""" Define number-partitioning Hamiltonian  """
def H_P(S):
    zeros = Qobj(np.zeros([2, 2]))
    H = zeros
    for i in range(len(S) - 1):
        H = tensor(H, zeros)
    for i, s in enumerate(S):
        H += s * z_gate(N=len(S), target=i)
    return pow(H, 2)

def H_B(S): # Mixing Hamiltonian
    zeros = Qobj(np.zeros([2, 2]))
    H = zeros
    for i in range(len(S) - 1):
        H = tensor(H, zeros)
    for i, s in enumerate(S):
        H += x_gate(N=len(S), target=i)
    return H
""" --------------------------------------- """

S = [5, 6, 7, 8] # Problem instance

VQA_circuit = VQA(
            n_qubits = len(S),
            n_layers = 5,
            cost_method = "OBSERVABLE",
        )
VQA_circuit.cost_observable = H_P(S)

VQA_circuit.add_block(
        VQA_Block(H_P(S), name="H_P")
        )
VQA_circuit.add_block(
        VQA_Block(H_B(S), name="H_B")
        )

# circuit initialisation with hadamards
for i in range(len(S)):
    VQA_circuit.add_block( 
            VQA_Block("SNOT", targets=[i], initial=True)
            )

result = VQA_circuit.optimize_parameters(
        method="BFGS",
        use_jac=True,
        initialization="random",
        )
result.plot(S)
print(result)
