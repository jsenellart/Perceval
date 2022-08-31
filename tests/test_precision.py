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

import pytest

from perceval import BackendFactory, BasicState
import perceval.components.base_components as comp

def test_precision_pcvl_124_enumerate():
    c = comp.SimpleBS()
    sim = BackendFactory().get_backend("Naive")(c.U)
    input_state = BasicState([20, 0])
    all_p = 0
    count_state = 0
    for output_state, p in sim.allstateprob_iterator(input_state):
        all_p += p
        count_state += 1
    assert count_state == 21
    assert pytest.approx(1) == all_p

def test_precision_pcvl_124_naive_manual():
    c = comp.SimpleBS()
    sim = BackendFactory().get_backend("Naive")(c.U)
    input_state = BasicState([20, 0])
    all_p = 0
    for i in range(21):
        all_p += sim.prob(input_state, BasicState([i, 20-i]))
    assert pytest.approx(1) == all_p
