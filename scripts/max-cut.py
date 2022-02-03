import numpy as np
from qutip import Qobj, tensor
from qutip_qip.operations import z_gate, x_gate
from qutip_qip.vqa import VQA, VQA_Block
import networkx as nx
import matplotlib.pyplot as plt
import random


"""
Define problem Hamiltonian in terms of a graph G = (V, E)
where E is a list of (u, v, w) for node indices u, v and weight w
"""
def H_P(V, E):
    n = len(V)
    # Initialize H
    zeros = Qobj(np.zeros([2, 2]))
    H = zeros
    for i in range(n - 1):
        H = tensor(H, zeros)
    # Add costs from each edge
    for u, v, w in E:
        H += 0.5 * w * (1 + z_gate(N=n, target=u) * z_gate(N=n, target=v))
    return H


def H_B(V):  # Mixing Hamiltonian
    n = len(V)
    zeros = Qobj(np.zeros([2, 2]))
    H = zeros
    for i in range(n - 1):
        H = tensor(H, zeros)
    for i in range(n):
        H += x_gate(n, target=i)
    return H


def brute_force(V, E):
    max_cut = 0
    opt_string = ""
    for i in range(2**len(V)):
        binary_string = bin(i)[2:].zfill(len(V))
        cut = bitstring_to_cut(binary_string, E)
        if cut > max_cut:
            max_cut = cut
            opt_string = binary_string
    return max_cut, opt_string


def bitstring_to_cut(bitstring, E):
    cut = 0
    for u, v, w in E:
        if bitstring[u] != bitstring[v]:
            cut += w
    return cut


"""
Problem Instance
----------------
Generate a random d-regular graph with n_nodes vertices
Note d * n_nodes must be even
"""
d = 3
n_nodes = 6
G = nx.generators.random_graphs.random_regular_graph(d, n_nodes)

V = list(G.nodes)
E = list(G.edges)
# add random weight to each edge
E = [(*e, random.random()) for e in E]


"""
Construct the VQA circuit and run it. Then compare the VQA's circuit's
highest probability measurement outcome against the brute force solution.
"""
VQA_circuit = VQA(
            n_qubits=len(V),
            n_layers=1,
            cost_method="OBSERVABLE",
        )

VQA_circuit.cost_observable = H_P(V, E)

# circuit initialisation with SNOT gates
for i in range(len(V)):
    VQA_circuit.add_block(
            VQA_Block("SNOT", targets=[i], initial=True)
            )

VQA_circuit.add_block(
        VQA_Block(H_P(V, E), name="H_P")
        )
VQA_circuit.add_block(
        VQA_Block(H_B(V), name="H_B")
        )

result = VQA_circuit.optimize_parameters(
        method="BFGS",
        use_jac=True,
        initial="random",
        layer_by_layer=False,
        )

bitstring = result.get_top_bitstring().strip('|').strip('>')
final_cut = bitstring_to_cut(bitstring, E)
brute_force_cut = brute_force(V, E)
approximation_ratio = round(final_cut / brute_force_cut[0], 2)

print(f"Approximation ratio: {approximation_ratio}")
"""
Plot results
"""
plot1 = plt.figure(1)
pos = nx.spring_layout(G)
edge_labels = {
    (u, v): round(w, 2) for u, v, w in list(E)
        }
node_pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos=pos)
nx.draw_networkx_labels(G, pos=pos)
nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, label_pos=.3)
plot2 = plt.figure(2)
result.plot(V, label_sets=True, top_ten=True)
