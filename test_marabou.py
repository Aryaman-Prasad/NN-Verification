import os
import re
import csv
import sys
import onnx
import argparse
import subprocess
import math
import argparse
import time
from maraboupy import Marabou
from vnnlib.compat import read_vnnlib_simple
import numpy as np
from pathlib import Path

##########################################################################################

# Parameters

ORIGINAL_ONNX = "vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x4.onnx"
APPENDED_ONNX = {}
for i in range(10):
    APPENDED_ONNX[i] = ORIGINAL_ONNX[:-5] + "_ext" + str(i) + ".onnx"

PROPERTY_FOLDER = "vnncomp2022_benchmarks/benchmarks/mnist_fc/vnnlib"

TIMEOUT = 60

RESULT_CSV = "results/results_mnist_fc_256x4_relaxed_robust_afzal_mara_50.csv"

##########################################################################################

class suppress_output:
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr

def get_counts(onnx_path):
    model = onnx.load(onnx_path)
    
    # Get the first input and output (filtering out weights)
    initializers = {n.name for n in model.graph.initializer}
    inputs = [i for i in model.graph.input if i.name not in initializers]
    output = model.graph.output[0]
    
    # Calculate total elements (ignore batch size if it's 0 or -1)
    in_dims = [d.dim_value for d in inputs[0].type.tensor_type.shape.dim if d.dim_value > 0]
    out_dims = [d.dim_value for d in output.type.tensor_type.shape.dim if d.dim_value > 0]
    
    return math.prod(in_dims), math.prod(out_dims)

# Property alteration, essentially take the original property and replace the output nodes with a single node whose constraint is >= 0.0

def get_property_label(vnnlib_path):
    """
    Extract label from the first comment line:
    ; Mnist property with label: i
    """

    with open(vnnlib_path, "r") as f:
        first_line = f.readline()

    match = re.search(r"label:\s*(\d+)", first_line)

    if match:
        return int(match.group(1))

    raise ValueError(f"Could not determine label from {vnnlib_path}")

def vnn_parser(onnx_file,vnn_file):

    num_inputs,num_outputs = get_counts(onnx_file)
    return read_vnnlib_simple(vnn_file,num_inputs,num_outputs)

def create_alt_property(original_path):
    """
    Create altered property:
    - Keep all X_i declarations and constraints
    - Keep only Y_0 declaration
    - Remove Y_1..Y_9 declarations
    - Replace final property with (assert (<= Y_0 0.0))
    """

    alt_path = original_path.replace(".vnnlib", "_alt.vnnlib")

    with open(original_path, "r") as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:

        # Stop when original property begins
        if "(assert (or" in line:
            break

        # Remove Y_1...Y_9 declarations
        if re.match(r"\(declare-const Y_[9] Real\)", line):
            continue

        new_lines.append(line)

    # Add new property
    new_lines.append("\n")

    # For 1st idea
    new_lines.append("(assert (<= Y_0 0.0))\n")

    # For 2nd idea
    # new_lines.append("(assert (<= Y_0 0.9))\n")

    # For afzal's idea
    # new_lines.append("(assert (or\n")
    # for idx in range(9):
    #     new_lines.append(f"    (and (>= Y_{idx} {-0.001}))\n")
    # new_lines.append("))\n")





    with open(alt_path, "w") as f:
        f.writelines(new_lines)

    return alt_path

# Running marabou

def run_marabou_once(input_bound, coeffs, output_bound : float, onnx_file):

    options = Marabou.createOptions(verbosity = 0,timeoutInSeconds=TIMEOUT)
    network = Marabou.read_onnx(onnx_file)
    inputVars = network.inputVars[0][0] 
    outputVars = network.outputVars[0]

    inputVars = np.ravel(inputVars)

    for i in range(len(input_bound)):
        network.setLowerBound(inputVars[i],input_bound[i][0])
        network.setUpperBound(inputVars[i],input_bound[i][1])

    network.addInequality(outputVars[0],coeffs,output_bound)

    with suppress_output():
        s = time.time()
        solution = network.solve(options=options)
        e = time.time()

    if solution is not None:
        if(solution[0]=='sat'): return "sat", e-s
        elif(solution[0]=='unsat'): return "unsat", e-s
        elif(solution[0]=='TIMEOUT'): return "timeout", e-s
        # for i in range(len(solution)):
        #     print(solution[i])
        #     print("\n \n")

    else:
        print("something has gone wrong")
        Exception("This was never supposed to happen")
        return "noooooo"

def run_marabou(onnx_path, property_path, mode):

    # bound processing
    bounds = vnn_parser(onnx_path, property_path)
    input_bounds = bounds[0][0] # list of arrays (lb ,ub) form
    output_bounds = bounds[0][1] # list of arrays (coeffs , ub)

    coeffs = []
    ubs = []

    for vars, ub in output_bounds:
        coeffs.append(vars[0])
        ubs.append(ub[0][0])


    sat = False
    tle = False

    time = 0.0

    for i in range(len(coeffs)):
        out, t = run_marabou_once(input_bounds, coeffs[i], ubs[i], onnx_path)
        time += t
        if (out == "sat"): 
            sat = True
            break
        if (time >= TIMEOUT): 
            tle = True
            break
    
    if sat: 
        return "sat", time
    elif tle : return "timeout", time
    else: return "unsat", time


# Main function for running an experiment

def run_experiment(mode):

    property_files = sorted(p for p in Path(PROPERTY_FOLDER).glob("*.vnnlib") if "_alt" not in p.name)

    results = []

    alt_files_created = []

    stats = {
        "unsat": 0,
        "sat": 0,
        "timeout": 0,
        "error": 0
    }

    for prop in property_files:

        prop = str(prop)

        if mode == "original":
            onnx_path = ORIGINAL_ONNX

        else:
            label = get_property_label(prop)
            onnx_path = APPENDED_ONNX[label]

            alt_prop = create_alt_property(prop)
            alt_files_created.append(alt_prop)
            alt_files_created.append(alt_prop + ".compiled")

            prop = alt_prop

        print(f"\nRunning property: {prop}")

        status, runtime = run_marabou(onnx_path, prop, mode)

        print("Result:", status, "Time:", runtime)

        stats[status] += 1

        results.append({
            "property": os.path.basename(prop),
            "result": status,
            "time": runtime
        })

    # Writing results into a csv

    with open(RESULT_CSV, "w", newline="") as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=["property", "result", "time"]
        )

        writer.writeheader()

        for row in results:
            writer.writerow(row)

    # Stats!

    total = sum(stats.values())

    print("\n===================================")
    print("Verification Summary")
    print("===================================")

    for k, v in stats.items():
        percent = 100 * v / total if total else 0
        print(f"{k}: {v} ({percent:.1f}%)")

    print("Total:", total)

    # Deletion of newly created altered properties

    for alt_file in alt_files_created:
        try:
            os.remove(alt_file)
        except OSError:
            pass


# Testing

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--original",
        action="store_true",
        help="Run original network verification"
    )

    parser.add_argument(
        "--appended",
        action="store_true",
        help="Run appended network verification"
    )

    args = parser.parse_args()

    if args.original:
        run_experiment("original")

    elif args.appended:
        run_experiment("appended")

    else:
        print("Specify --original or --appended")

    # An out.txt gets created by abcrown, deleting it

    try:
        os.remove("out.txt")
    except OSError:
        pass