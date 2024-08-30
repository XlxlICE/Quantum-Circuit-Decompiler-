# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Compute the sum of two equally sized qubit registers."""
from __future__ import annotations
import numpy as np
import random
from fractions import Fraction
from qiskit import AncillaRegister,ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT
from qiskit.algorithms import Grover
from qiskit.circuit.library import GroverOperator
from math import pi

def h_c(n) -> QuantumCircuit:
    
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(i)
    return qc



def h_0(n):
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(0)
    return qc

def ry_c(n):
    qc = QuantumCircuit(n)
    angle = pi
    for i in range(n):
        qc.ry(angle, i)
        angle /= 2
    return qc

def ry_decomposed(n):
    qc = QuantumCircuit(n)
    angle = pi
    for i in range(n):
        qc.h(i)  # Apply H gate
        qc.rx(angle, i)  # Apply RX gate
        qc.h(i)  # Apply H gate
        angle /= 2
    return qc


def ry_decomposed_rx_rz(n):
    qc = QuantumCircuit(n)
    angle = pi
    for i in range(n):
        qc.rz(-pi/2, i)  # Apply RZ gate
        qc.rx(angle, i)  # Apply RX gate
        qc.rz(pi/2, i)  # Apply RZ gate
        angle /= 2
    return qc

def rx_c(n):
    qc = QuantumCircuit(n)
    angle = pi
    for i in range(n):
        qc.rx(angle, i)
        angle /= 2
    return qc

def rx_gradually_c(n):
    qc = QuantumCircuit(n)
    angle = pi
    for level in range(1, n + 1):
        current_angle = angle / (2 ** (level - 1))
        for i in range(level):
            qc.rx(current_angle, i)
    return qc

def qpe(num_qubits: int) -> QuantumCircuit:
    """Returns a quantum circuit implementing the Quantum Phase Estimation algorithm for a phase which can be
    exactly estimated.

    Keyword arguments:
    num_qubits -- number of qubits of the returned quantum circuit
    """

    num_qubits = num_qubits - 1  # because of ancilla qubit
    q = QuantumRegister(num_qubits, "q")
    psi = QuantumRegister(1, "psi")
    c = ClassicalRegister(num_qubits, "c")
    qc = QuantumCircuit(q, psi, c, name="qpeexact")

    # get random n-bit string as target phase
    random.seed(10)
    theta = 0
    while theta == 0:
        theta = random.getrandbits(num_qubits)
    lam = Fraction(0, 1)
    # print("theta : ", theta, "correspond to", theta / (1 << n), "bin: ")
    for i in range(num_qubits):
        if theta & (1 << (num_qubits - i - 1)):
            lam += Fraction(1, (1 << i))

    qc.x(psi)
    qc.h(q)

    for i in range(num_qubits):
        angle = (lam * (1 << i)) % 2
        if angle > 1:
            angle -= 2
        if angle != 0:
            qc.cp(angle * np.pi, psi, q[i])

    qc.compose(
        QFT(num_qubits=num_qubits, inverse=True),
        inplace=True,
        qubits=list(range(num_qubits)),
    )
    qc.barrier()
    qc.measure(q, c)

    return qc

def qft(num_qubits: int) -> QuantumCircuit:
    """Returns a quantum circuit implementing the Quantum Fourier Transform algorithm.

    Keyword arguments:
    num_qubits -- number of qubits of the returned quantum circuit
    """

    q = QuantumRegister(num_qubits, "q")
    c = ClassicalRegister(num_qubits, "c")
    qc = QuantumCircuit(q, c, name="qft")
    qc.compose(QFT(num_qubits=num_qubits), inplace=True)
    qc.measure_all()

    return qc

def qft_dec(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
    # Apply the Hadamard gate to the i-th qubit
        qc.h(i)
    # Apply controlled phase rotation gates
    for i2 in range(num_qubits - i - 1):  # i2 is now the relative index from i+1
        # Compute the actual index of the qubit to interact with
        target_qubit = i + i2 + 1
        angle = np.pi / (2 ** (i2 + 1))  # Adjusted to reflect the relative position
        qc.cp(angle, target_qubit, i)
    
def qft_decom(num_qubits: int) -> QuantumCircuit:
    """Returns a quantum circuit implementing the decomposed Quantum Fourier Transform.

    Keyword arguments:
    num_qubits -- number of qubits of the returned quantum circuit
    """
    qc = QuantumCircuit(num_qubits, name="Decomposed QFT")

    # Apply the Hadamard and controlled phase gates
    for j in range(num_qubits):
        # Apply the Hadamard gate to the j-th qubit
        qc.h(j)
        # Apply controlled phase rotation gates
        for k in range(j+1, num_qubits):
            angle = np.pi / (2 ** (k - j))
            qc.cp(angle, k, j)

    # Swap qubits to reverse the order of outputs
    for j in range(num_qubits // 2):
        qc.swap(j, num_qubits-j-1)

    return qc

def qpe_dec(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits + 1)
    
    # Apply Hadamard gates to the first 'num_qubits' qubits
    for i in range(num_qubits):
        qc.h(i)
    
    # Apply controlled unitary operations
    for i in range(num_qubits):
        for j in range(2**i):
            qc.cp(np.pi / 2**i, num_qubits - 1 - i, num_qubits)
    
    # Apply inverse QFT to the first 'num_qubits' qubits
    qc.append(qft_dec(num_qubits).inverse(), range(num_qubits))
    
    return qc

def grover(num_qubits: int, ancillary_mode: str = "noancilla") -> QuantumCircuit:
    """Returns a quantum circuit implementing Grover's algorithm.

    Keyword arguments:
    num_qubits -- number of qubits of the returned quantum circuit
    ancillary_mode -- defining the decomposition scheme
    """

    num_qubits = num_qubits - 1  # -1 because of the flag qubit
    q = QuantumRegister(num_qubits, "q")
    flag = AncillaRegister(1, "flag")

    state_preparation = QuantumCircuit(q, flag)
    state_preparation.h(q)
    state_preparation.x(flag)

    oracle = QuantumCircuit(q, flag)
    oracle.mcp(np.pi, q, flag)

    operator = GroverOperator(oracle, mcx_mode=ancillary_mode)
    iterations = Grover.optimal_num_iterations(1, num_qubits)

    num_qubits = operator.num_qubits - 1  # -1 because last qubit is "flag" qubit and already taken care of

    # num_qubits may differ now depending on the mcx_mode
    q2 = QuantumRegister(num_qubits, "q")
    qc = QuantumCircuit(q2, flag, name="grover")
    qc.compose(state_preparation, inplace=True)

    qc.compose(operator.power(iterations), inplace=True)
    qc.measure_all()
    qc.name = qc.name + "-" + ancillary_mode

    return qc


def ghz_state(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc

def qpe_dec(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits + 1)
    
    # Apply Hadamard gates to the first 'num_qubits' qubits
    for i in range(num_qubits):
        qc.h(i)
    
    # Apply controlled unitary operations
    for i in range(num_qubits):
        for j in range(2**i):
            qc.cp(np.pi / 2**i, num_qubits - 1 - i, num_qubits)
    
    # Apply inverse QFT to the first 'num_qubits' qubits
    for i in range(num_qubits // 2):
        qc.swap(i, num_qubits - i - 1)
    
    for i in range(num_qubits):
        qc.h(i)
        for j in range(i):
            qc.cp(np.pi / 2**(i-j), j, i)
    
    return qc


