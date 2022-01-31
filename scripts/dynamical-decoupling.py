import numpy as np
from qutip import Qobj, tensor
from qutip_qip.operations import z_gate, x_gate
from qutip_qip.vqa import VQA, VQA_Block

H = Qobj([
    [1, 1],
    [1, -1]
    ])

k = 1
VQA_circuit = VQA(
            n_qubits = 1,
            n_layers = k,
            cost_method = "OBSERVABLE",
        )

U_block = VQA_Block(1/(2*k) * H, name="U(t)")
U_block = VQA_Block(Qobj([[0, 1],[1, 0]]),name="X(t)")
VQA_circuit.add_block(U_block)
VQA_circuit.add_block(VQA_Block("x_gate", targets=[0]))
VQA_circuit.repeat_block(U_block)
VQA_circuit.add_block(VQA_Block("z_gate", targets=[0]))
VQA_circuit.export_image()
print(VQA_circuit.get_final_state([0.5]))
