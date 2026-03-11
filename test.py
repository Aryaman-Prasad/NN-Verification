import os
import re
import csv
import argparse
import subprocess
import yaml
import tempfile
from pathlib import Path

##########################################################################################

# Parameters

ORIGINAL_ONNX = "vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x2.onnx"
APPENDED_ONNX = {
    0: "vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x2_ext0.onnx",
    1: "vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x2_ext1.onnx",
    2: "vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x2_ext2.onnx",
    3: "vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x2_ext3.onnx",
    4: "vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x2_ext4.onnx",
    5: "vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x2_ext5.onnx",
    6: "vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x2_ext6.onnx",
    7: "vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x2_ext7.onnx",
    8: "vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x2_ext8.onnx",
    9: "vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x2_ext9.onnx",
}

PROPERTY_FOLDER = "vnncomp2022_benchmarks/benchmarks/mnist_fc/vnnlib"

ABCROWN_SCRIPT = "alpha-beta-CROWN/complete_verifier/abcrown.py"

TIMEOUT = 300

RESULT_CSV = "results_mnist_fc.csv"

##########################################################################################

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


def create_alt_property(original_path):
    """
    Create altered property:
    - Keep all X_i declarations and constraints
    - Keep only Y_0 declaration
    - Remove Y_1..Y_9 declarations
    - Replace final property with (assert (>= Y_0 0.0))
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
        if re.match(r"\(declare-const Y_[1-9] Real\)", line):
            continue

        new_lines.append(line)

    # Add new property
    new_lines.append("\n")
    new_lines.append("(assert (>= Y_0 0.0))\n")

    with open(alt_path, "w") as f:
        f.writelines(new_lines)

    return alt_path

# Running alpha-beta-crown

def run_abcrown(onnx_path, property_path):

    with open("test.yaml") as f:
        config = yaml.safe_load(f)

    config["model"]["onnx_path"] = onnx_path
    config["specification"]["vnnlib_path"] = property_path
    config["bab"]["timeout"] = TIMEOUT

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:

        yaml.dump(config, tmp)
        config_path = tmp.name

    cmd = [
        "python",
        ABCROWN_SCRIPT,
        "--config",
        config_path
    ]

    try:

        result = subprocess.run(cmd, capture_output=True, text=True)

        output = result.stdout + result.stderr

        status = "error"
        runtime = None

        if "Result: unsat" in output:
            status = "unsat"
        elif "Result: sat" in output:
            status = "sat"
        elif "timeout" in output.lower():
            status = "timeout"

        match = re.search(r"Time:\s*([0-9\.]+)", output)

        if match:
            runtime = float(match.group(1))

        return status, runtime

    except Exception:
        return "error", None


# Main function for running an experiment

def run_experiment(mode):

    property_files = sorted(Path(PROPERTY_FOLDER).glob("*.vnnlib"))

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

        status, runtime = run_abcrown(onnx_path, prop)

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