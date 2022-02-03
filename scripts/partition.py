import numpy as np
from qutip import Qobj, tensor
from qutip_qip.operations import z_gate, x_gate
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


def H_B(S):  # Mixing Hamiltonian
    zeros = Qobj(np.zeros([2, 2]))
    H = zeros
    for i in range(len(S) - 1):
        H = tensor(H, zeros)
    for i, s in enumerate(S):
        H += x_gate(N=len(S), target=i)
    return H


""" --------------------------------------- """

def brute_force(S):
    s0 = S
    s1 = []
    best_s0 = []
    best_s1 = []
    best_cost = np.inf
    for i in range(2**len(S)):
        binary_string = bin(i)[2:].zfill(len(S))
        s0 = [S[i] for i in range(len(S)) if binary_string[i] == '0']
        s1 = [S[i] for i in range(len(S)) if binary_string[i] == '1']
        cost = abs(sum(s0) - sum(s1))
        if cost < best_cost:
            best_cost = cost
            best_s0 = s0
            best_s1 = s1
    print(f'Best cost: {best_cost}, s0: {best_s0}, s1: {best_s1}')


s = [1, 4, 3] # Problem instance
s = [48, 52, 60, 39, 6]

VQA_circuit = VQA(
            n_qubits=len(s),
            n_layers=20,
            cost_method="OBSERVABLE",
        )
VQA_circuit.cost_observable = H_P(s)

# circuit initialisation with SNOT gates
for i in range(len(s)):
    VQA_circuit.add_block(
            VQA_Block("SNOT", targets=[i], initial=True)
            )

VQA_circuit.add_block(
        VQA_Block(H_P(s), name="H_P")
        )
VQA_circuit.add_block(
        VQA_Block(H_B(s), name="H_B")
        )

print("Running optimization...")
result = VQA_circuit.optimize_parameters(
        method="BFGS",
        use_jac=True,
        initial="random",
        layer_by_layer=True,
        )

print("Brute force solution:")
brute_force(s)
result.plot(s, label_sets=True, top_ten=True)
print(result)
