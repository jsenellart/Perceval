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

import perceval as pcvl
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Optional
import perceval.components.base_components as comp
from tqdm import tqdm

# definition of the circuit
C = pcvl.Circuit(2)
C.add((0, 1), comp.SimpleBS())
C.add((1), comp.TD(1))
C.add((0, 1), comp.SimpleBS())

# definition of the source - it is a probabilistic mixture of statevectors
source = pcvl.SVDistribution()
# first example, perfect source: 100% probability of emitting one photon
#source[pcvl.StateVector(pcvl.BasicState([1,0]))] = 1

# second example: brightness 100%, but each statevector is superposed 0.95*|1,0>+0.05*|0,0>
source[pcvl.StateVector(pcvl.BasicState([1,0]))*0.95+pcvl.StateVector(pcvl.BasicState([0,0]))*0.05] = 0.6

# last example: perfect source with a brightness of 60%
#source[pcvl.StateVector(pcvl.BasicState([1,0]))] = 0.60
#source[pcvl.StateVector(pcvl.BasicState([0,0]))] = 0.40



stepper = pcvl.BackendFactory().get_backend("Stepper")(C)


def apply_dt(sv: pcvl.StateVector, k: int, nk: int) -> pcvl.StateVector:
    nsv = pcvl.StateVector()
    for state in sv:
        n_state = list(copy.copy(state))
        n_state[k] = nk
        nsv[pcvl.BasicState(n_state)] = sv[state]
    return nsv


dist_deltat = defaultdict(lambda: 0)
def click(last_event, t, state, prob):
    global dist_deltat
    if state.n == 0:
        return last_event
    if state[0] > 0 and state[1] > 0:
        dist_deltat[0] += prob
    elif last_event[1] is not None and state[last_event[1]] == 0:
        delta = t - last_event[0]
        if state[0]:
            delta = -delta
        dist_deltat[delta] += prob
    if state[0] > 0:
        k_click = 0
    else:
        k_click = 1
    return (t, k_click, prob)


pbar = tqdm(total=1)

rec_level = 0

cache = {}


def rec_run_clicks(t, idx, clicks, last_event):
    while idx < len(clicks) and clicks[idx][0] > t:
        sub_last_event = click(last_event, clicks[idx][0], clicks[idx][1], clicks[idx][2])
        idx = rec_run_clicks(clicks[idx][0], idx+1, clicks, sub_last_event)
    return idx


def run(circuit, sv: Optional[pcvl.StateVector], source: pcvl.SVDistribution, waiting_photons,
        t, max_t, c_start=None, prob=1, last_event=None):
    global cache
    key = None
    if c_start is None:
        key = str(sv)+"--waiting:"+",".join([str(k)+":"+str(v) for k,v in waiting_photons[t].items()])
        if t in cache and key in cache[t]:
            # we already have this complete sequence, we can just run it from the cache
            prob_cache, clicks = cache[t][key]
            # adjust clicks probability
            clicks = [(t, s, p * prob / prob_cache) for (t, s, p) in clicks]
            rec_run_clicks(t-1, 0, clicks, last_event)
            return clicks
    clicks = []
    # returns the time-sequence of click it generates
    if sv is None:
        # we are necessarily at the beginning of the circuit
        assert c_start is None
        if t-int(t) < 1e-6:
            for sv, prob_sv in source.items():
                clicks += run(circuit, sv, source, waiting_photons,
                              t, max_t, c_start, prob_sv*prob, last_event)
            return clicks
        else:
            return run(circuit, pcvl.StateVector(pcvl.BasicState(circuit.m)), source, waiting_photons,
                t, max_t, c_start, prob, last_event)

    global rec_level
    rec_level += 1

    # finding position in circuit iterator
    c_iter = circuit.__iter__()
    if c_start is not None:
        while True:
            (_, c) = c_iter.__next__()
            if id(c) == c_start:
                break

    try:
        while True:
            (r, c) = c_iter.__next__()
            if not c.delay_circuit:
                sv = stepper.apply(sv, r, c)
            else:
                delta_t = float(c._dt)
                if delta_t > 0:
                    # we need to inject possibly waiting photons here
                    k = r[0]
                    # deploy all possible extensions
                    for state, pa in sv.items():
                        nsv = pcvl.StateVector()
                        nsv[state] = sv[state]
                        nsv = apply_dt(nsv, k, waiting_photons[t][id(c)])
                        waiting_photons[t + delta_t][id(c)] = state[k]
                        # run the following of the circuit recursively, starting again after the delay
                        clicks += run(circuit, nsv, source, waiting_photons, t, max_t, id(c), prob*abs(pa)**2, last_event)
                    rec_level -= 1
                    if key is not None:
                        if t not in cache:
                            cache[t] = {}
                        cache[t][key] = (prob, copy.copy(clicks))
                    return clicks
    except StopIteration:
        # at the end of the circuit - current state vector is representing the t-state
        pass
    # check for the next event looking at waiting_photons
    next_timesteps = [timestep for timestep in waiting_photons.keys() if timestep > t]
    next_timesteps.append(t + 1)
    next_t = min(next_timesteps)

    for state in sv:
        prob_sv = abs(sv[state]) ** 2 * prob
        if t == 1:
            pbar.update(prob_sv)
        if t >= 2:
            clicks.append((t, state, prob))
            new_last_event = click(last_event, t, state, prob)
        else:
            new_last_event = last_event
        if next_t <= max_t:
            clicks += run(circuit, None, source, waiting_photons, next_t, max_t, None, prob_sv, new_last_event)
    rec_level -= 1
    return clicks

next_waiting_photons = defaultdict(lambda: defaultdict(lambda: 0))
run(C, None, source, next_waiting_photons, 0, 8, None, 1, (None, None, 1))
pbar.close()

print(dist_deltat)

delays = list(dist_deltat.keys())
clicks = [dist_deltat[d] for d in delays]

plt.bar(delays, clicks, color="green")

plt.xlabel("Delays")
plt.ylabel("Clicks")

plt.plot()
plt.show()
