import itertools
import subprocess
import os

# Parameters
L1Sizes = ["16kB", "32kB", "64kB"]
L2Sizes = ["128kB", "256kB", "512kB"]
L3Sizes = ["0kB", "2MB", "4MB"]

GEM5 = "gem5/build/X86/gem5.opt"
# TODO
SCRIPT = "simulate.py" 

OUTROOT = "./finalProject/generic"

# Run all combinations
for l1, l2, l3 in itertools.product(L1Sizes, L2Sizes, L3Sizes):

    # Output dir
    outputDir = f"{OUTROOT}/L1_{l1}_L2_{l2}_L3_{l3}"
    os.makedirs(outputDir, exist_ok=True)

    # Assemble command
    cmd = [
        GEM5,
        "-d", outputDir,
        SCRIPT,
        "--l1i_size", l1,
        "--l1d_size", l1,
        "--l2_size", l2,
        ]
    
    if l3 != "0kB":
        cmd += ["--l3_size", l3]

    print("Running permutation: ", cmd)
    subprocess.run(cmd)
