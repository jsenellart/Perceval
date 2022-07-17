# MIT License
#
# Copyright (c) 2022 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import pytest

try:
    import qiskit
except Exception as e:
    pytest.skip("need `qiskit` module", allow_module_level=True)

from perceval.converters import perceval_qiskit
import perceval.lib.phys as phys


def _check_perm(perm, c_perm):
    u = c_perm.U
    for v, i in enumerate(perm):
        assert u[i, v] == 1


def test_basic_circuit_h():
    qc = qiskit.QuantumCircuit(1)
    qc.h(0)
    pc = perceval_qiskit.to_perceval(qc, phys)
    c = pc.circuit
    sources = pc._sources
    assert len(sources) == 1
    assert 0 in sources
    assert 1 not in sources
    assert len(c._components) == 1
    assert isinstance(c._components[0][1], phys.Circuit) and len(c._components[0][1]._components) == 1
    c0 = c._components[0][1]._components[0][1]
    assert isinstance(c0, phys.BS)


def test_basic_circuit_doubleh():
    qc = qiskit.QuantumCircuit(1)
    qc.h(0)
    qc.h(0)
    pc = perceval_qiskit.to_perceval(qc, phys)
    c = pc.circuit
    sources = pc._sources
    assert len(sources) == 1
    assert 0 in sources
    assert 1 not in sources
    assert len(c._components) == 2


def test_basic_circuit_s():
    qc = qiskit.QuantumCircuit(1)
    qc.s(0)
    pc = perceval_qiskit.to_perceval(qc, phys)
    c = pc.circuit
    sources = pc._sources
    assert len(sources) == 1
    assert 0 in sources
    assert 1 not in sources
    assert len(c._components) == 1
    assert isinstance(c._components[0][1], phys.Circuit) and len(c._components[0][1]._components) == 1
    r0 = c._components[0][1]._components[0][0]
    c0 = c._components[0][1]._components[0][1]
    assert r0 == (1, )
    assert isinstance(c0, phys.PS)


def test_basic_circuit_swap_direct():
    qc = qiskit.QuantumCircuit(2)
    qc.swap(0, 1)
    pc = perceval_qiskit.to_perceval(qc, phys)
    c = pc.circuit
    sources = pc._sources
    assert len(sources) == 2
    assert 0 in sources
    assert 1 not in sources
    assert 2 in sources
    assert 3 not in sources
    assert len(c._components) == 1
    r0, c0 = c._components[0]
    assert r0 == [0, 1, 2, 3]
    assert isinstance(c0, phys.PERM)
    _check_perm([2,3,0,1], c0)


def test_basic_circuit_swap_indirect():
    qc = qiskit.QuantumCircuit(2)
    qc.swap(1, 0)
    pc = perceval_qiskit.to_perceval(qc, phys)
    c = pc.circuit
    sources = pc._sources
    assert len(sources) == 2
    assert 0 in sources
    assert 2 in sources
    assert len(c._components) == 1
    r0, c0 = c._components[0]
    assert r0 == [0, 1, 2, 3]
    assert isinstance(c0, phys.PERM)
    _check_perm([2,3,0,1], c0)


def test_cnot_1():
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    pc = perceval_qiskit.to_perceval(qc, phys)
    c = pc.circuit
    sources = pc._sources
    assert len(sources) == 4
    assert 0 in sources and 2 in sources and 5 in sources and 6 in sources
    assert len(c._components) == 4
    # should be BS//PERM//CNOT//PERM
    perm1 = c._components[1][1]
    assert isinstance(perm1, phys.PERM)
    perm2 = c._components[3][1]
    assert isinstance(perm2, phys.PERM)
    # check that ports are correctly connected
    _check_perm([2,3,4,5,0,1,6,7], perm1)
    _check_perm([4,5,0,1,2,3,6,7], perm2)


def test_cnot_1_inverse():
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(1, 0)
    pc = perceval_qiskit.to_perceval(qc, phys)
    c = pc.circuit
    sources = pc._sources
    assert len(sources) == 4
    assert 0 in sources and 2 in sources and 5 in sources and 6 in sources
    assert len(c._components) == 4
    # should be BS//PERM//CNOT//PERM
    perm1 = c._components[1][1]
    assert isinstance(perm1, phys.PERM)
    perm2 = c._components[3][1]
    assert isinstance(perm2, phys.PERM)
    # check that ports are correctly connected
    _check_perm([4,5,2,3,0,1,6,7], perm1)
    _check_perm([4,5,2,3,0,1,6,7], perm2)


def test_cnot_2():
    qc = qiskit.QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 2)
    pc = perceval_qiskit.to_perceval(qc, phys)
    c = pc.circuit
    sources = pc._sources
    assert len(sources) == 5
    assert 0 in sources and 2 in sources and 4 in sources and 7 in sources and 8 in sources
    assert len(c._components) == 4
    # should be BS//PERM//CNOT//PERM
    perm1 = c._components[1][1]
    assert isinstance(perm1, phys.PERM)
    perm2 = c._components[3][1]
    assert isinstance(perm2, phys.PERM)
    # check that ports are correctly connected
    _check_perm([2,3,8,9,4,5,0,1,6,7], perm1)
    _check_perm([6,7,0,1,4,5,8,9,2,3], perm2)
