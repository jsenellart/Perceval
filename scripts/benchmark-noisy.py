#!/usr/bin/env python
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

# coding: utf-8

from perceval.utils.statevector import global_params
import perceval as pcvl
import time
import argparse
import numpy as np
from copy import copy
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--m', type=int, required=True, help='number of modes')
parser.add_argument('--n', type=int, required=True, help='number of photons')
parser.add_argument('--precision', type=int, default=16, help='precision in calculation - number of digit')
parser.add_argument('--adaptive_precision', action="store_true", default=False, help='if set, use precision as a offset'
                                                                                     'on the class max prob')
parser.add_argument('--max_coincidence', type=int, default=1, help='maximum coincidence order we are interested in'
                                                                   'relatively to number of photons'
                                                                   ' (default +1)')
parser.add_argument('--algorithm', type=str, required=True)
parser.add_argument('--mode_selection', type=int, default=0)
parser.add_argument('--threshold', action="store_true", default=False, help='use for threshold detector')
parser.add_argument('--output_dir', default="", help='directory to save outputs, by default does not save output')
parser.add_argument('--loss', type=float, default=0.91, help='loss to apply on all channels')
args = parser.parse_args()

with open(f"tests/data/random_unitary_{args.m}", "rb") as f:
    unitary = pcvl.MatrixN(np.load(f))

global_params['min_p'] = 10**(-args.precision)

start = time.time()
source = pcvl.Source(multiphoton_component=0.01, indistinguishability=0.9, emission_probability=0.85, losses=args.loss)
QPU = pcvl.Processor("SLOS", args.m, source)
QPU.add(0, pcvl.Unitary(unitary))
QPU.min_detected_photons_filter(args.mode_selection)
QPU.with_input(pcvl.BasicState([1]*args.n+[0]*(args.m-args.n)))
QPU.thresholded_output(args.threshold)
QPU._setup_simulator()
print("time for input initialization...", time.time()-start)

def initial_algorithm_prob(simulator, input_state, output_state, skip_compile):
    input_states = input_state.separate_state()
    all_prob = 0
    for p_output_state in pcvl.BasicState(output_state).partition(
            [input_state.n for input_state in input_states]):
        prob = 1
        for i_state, o_state in zip(input_states, p_output_state):
            if not skip_compile:
                simulator.compile(i_state)
            prob *= simulator.prob_be(i_state, o_state)
        all_prob += prob
    return all_prob

def initial_algorithm_prob_iterator(simulator, input_state):
    skip_compile=False
    for output_state in simulator.allstate_iterator(input_state):
        if isinstance(input_state, pcvl.StateVector):
            input_state = input_state[0]
        yield output_state, initial_algorithm_prob(simulator, input_state, output_state, skip_compile=skip_compile)
        skip_compile = True

def initial_algorithm_probs():
    output = pcvl.BSDistribution()
    p_logic_discard = 0
    physical_perf = 1
    pbar = tqdm(total=len(QPU._inputs_map), smoothing=0)
    for idx, (input_state, input_prob) in enumerate(QPU._inputs_map.items()):
        if not QPU._state_preselected_physical(input_state):
            physical_perf -= input_prob
        else:
            for (output_state, p) in initial_algorithm_prob_iterator(QPU._simulator, input_state):
                if p < global_params['min_p']:
                    continue
                output_prob = p * input_prob
                if not QPU._state_selected_physical(output_state):
                    physical_perf -= output_prob
                    continue
                if QPU._state_selected(output_state):
                    output[QPU.postprocess_output(output_state)] += output_prob
                else:
                    p_logic_discard += output_prob
        pbar.update(1)
    pbar.close()
    return p_logic_discard, physical_perf, output

from collections import Counter, defaultdict

def improved_algorithm_register_probs(n, QPU, current_prob, input_states, output_state, class_max_prob):
    output_distribution = Counter()
    if current_prob < global_params['min_p']:
        # abandon this decomposition, we are already below the precision threshold
        return 0, output_distribution, True
    if not input_states:
        # there are no more components in input_states
        if current_prob > class_max_prob[n]:
            class_max_prob[n] = current_prob
            if args.adaptive_precision:
                global_params['min_p'] = 10 ** (-args.precision) * current_prob
        if not QPU._state_selected_physical(output_state):
            # it is not selected
            return 0, output_distribution, False
        if QPU._state_selected(output_state):
            output_distribution[QPU.postprocess_output(pcvl.BasicState(output_state))] = current_prob
            return 0, output_distribution, False
        else:
            return current_prob, output_distribution, False
    else:
        p_logic_discard = 0
        # build all possible outputs for the given input
        input_state = input_states[0]
        truncated = False
        for (o_state, p) in initial_algorithm_prob_iterator(QPU._simulator, input_state):
            new_output_state = copy(output_state)
            for idx, k in enumerate(o_state):
                new_output_state[idx] += k
            o_p_logic_discard, o_output_distribution, o_truncated = \
                improved_algorithm_register_probs(n, QPU, current_prob*p,
                                                  input_states[1:], new_output_state, class_max_prob)
            p_logic_discard += o_p_logic_discard
            output_distribution += o_output_distribution
            truncated = truncated or o_truncated
        return p_logic_discard, output_distribution, truncated

p_exp = 0

def improved_algorithm_register_combine(n, m, QPU, current_prob, output_distribution, pc_output_to_combine,
                                        current_output, max_p, class_max_prob):
    global p_exp
    p_exp += 1
    if current_prob < global_params['min_p']:
        # abandon this decomposition, we are already below the precision threshold
        return 0
    if not pc_output_to_combine:
        if current_prob > class_max_prob[n]:
            class_max_prob[n] = current_prob
        if not QPU._state_selected_physical(current_output):
            # it is not selected
            return 0
        if QPU._state_selected(current_output):
            output_distribution[QPU.postprocess_output(pcvl.BasicState(current_output))] += current_prob
            return 0
        else:
            return current_prob

    logical_perf = 0
    first_c_p = pc_output_to_combine[0]
    # is any of the following combination can give us a result
    if max_p[0] * current_prob < global_params['min_p']:
        return 0
    for (o, p) in first_c_p['outputs']:
        for idx, v in o.items():
            current_output[idx] += v
        logical_perf += improved_algorithm_register_combine(n, m, QPU, current_prob*p, output_distribution,
                                                            pc_output_to_combine[1:], current_output, max_p[1:], class_max_prob)
        for idx, v in o.items():
            current_output[idx] -= v
    return logical_perf


def unique_and_sort(m, inputs_map):
    sv_dist = defaultdict(lambda: 0)
    n_elem_class = Counter()
    for k, p in inputs_map.items():
        annot_map={}
        s=[""]*m
        for i in range(k[0].n):
            mode = k[0].photon2mode(i)
            annot = k[0].get_photon_annotation(i)
            if annot_map.get(str(annot)) is None:
                annot_map[str(annot)]="{a:%d}" % len(annot_map)
            s[mode] += annot_map[str(annot)]
        state=pcvl.StateVector("|"+",".join([v and v or "0" for v in s])+">")
        if state not in sv_dist:
            n_elem_class[k[0].n] += 1
        sv_dist[state] += p
    return sorted(sv_dist.items(), key=lambda x: -x[1]), n_elem_class

dcombine = 0
lcombine = 0
ncombine = 0

def improved_algorithm_probs():
    global dcombine, lcombine, ncombine
    output_distribution = Counter()
    logical_perf = 0
    class_max_prob = {}
    for i in range(2*args.n+1):
        if i - args.n > args.max_coincidence:
            class_max_prob[i] = 1
        else:
            class_max_prob[i] = 0
    pbar = tqdm(total=len(QPU._inputs_map), smoothing=0)

    sv_distribution, n_elem_class = unique_and_sort(args.m, QPU._inputs_map)
    avg_time = Counter()

    cache_output = defaultdict(lambda:{'max_p': 0, 'outputs': []})

    for idx, (input_state, input_prob) in enumerate(sv_distribution):
        if args.adaptive_precision:
            global_params['min_p'] = 10 ** (-args.precision) * class_max_prob[input_state[0].n]
        if not QPU._state_preselected_physical(input_state) or input_prob < global_params['min_p']:
            pass
        else:
            start_loop = time.time()
            if args.adaptive_precision:
                global_params['min_p'] = 10**(-args.precision)*class_max_prob[min(args.n, input_state[0].n)]/n_elem_class[input_state[0].n]
            input_state_components = input_state[0].separate_state()
            # we have a list of unannotated input_state components that can generate many output states that will
            # combine to make global output state
            c_output_to_combine = []
            for component in input_state_components:
                max_p_component_outputs = cache_output[component]
                if not max_p_component_outputs["outputs"]:
                    for (o_state, p) in initial_algorithm_prob_iterator(QPU._simulator, component):
                        max_p_component_outputs["outputs"].append((({i:k for i,k in enumerate(list(o_state)) if k}), p))
                        if p > max_p_component_outputs["max_p"]:
                            max_p_component_outputs["max_p"] = p
                c_output_to_combine.append(max_p_component_outputs)
            max_p = [c_output_to_combine[-1]["max_p"]]*len(c_output_to_combine)
            for k in range(2, len(c_output_to_combine)):
                max_p[-k] = c_output_to_combine[-k]["max_p"] * max_p[-k+1]
            # now we just need to combine these output states while preserving the truncated probability
            input_logic_discard = improved_algorithm_register_combine(input_state[0].n, args.m, QPU, input_prob,
                                                                      output_distribution, c_output_to_combine,
                                                                      [0]*args.m, max_p, class_max_prob)
            logical_perf += input_logic_discard
            ncombine += 1
            lcombine+=len(c_output_to_combine)
            dcombine+=sum([len(m["outputs"]) for m in c_output_to_combine])/len(c_output_to_combine)
            avg_time[input_state[0].n]+=time.time()-start_loop
        pbar.update(1)
    pbar.close()
    physical_perf = sum([v for v in output_distribution.values()])-logical_perf
    print("time:\n","".join(["%d\t%d\t%f\n" % (n, avg_time[n], avg_time[n]/n_elem_class[n]) for n in range(len(n_elem_class))]))
    print(ncombine, dcombine/ncombine, lcombine/ncombine)
    os.exit()
    return logical_perf, physical_perf, output_distribution

def current_algorithm_probs():
    results = pcvl.algorithm.Sampler(QPU).probs()
    return results["logical_perf"], results["physical_perf"], results["results"]

start = time.time()
logical_perf, physical_perf, output_distribution = locals()[args.algorithm+"_algorithm_probs"]()
end = time.time()
print("time for simulation...", end-start)

print("p_exp", p_exp)

print("logical perf", logical_perf, "physical perf", physical_perf)

if args.output_dir:
    max_coincidence = args.max_coincidence != 1 and f"_{args.max_coincidence}" or ""
    filename = (f"sim{args.m}:{args.n}-{args.adaptive_precision and 'P' or 'p'}{args.precision}-{args.algorithm}++"
                f"-ms{args.mode_selection}{max_coincidence}-{args.threshold and 'ts' or 'pnr'}-loss{args.loss}")
    with open(args.output_dir+"/"+filename+".txt", "w") as fw:
        for k, v in output_distribution.items():
            fw.write(f"{str(k)}\t{v}\n")
    with open(args.output_dir+"/"+filename+".log", "w") as fw:
        fw.write(f"time\t{end-start}\n")
        fw.write(f"logical_perf\t{logical_perf}\n")
        fw.write(f"physical_perf\t{physical_perf}\n")
    print("out/log files: ", filename+".txt/log")
