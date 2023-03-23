import sys

from analysis.experiments import fix_artifact_paths


if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) != 2:
        raise ValueError("Please provide the path to the mlruns folder as an argument.")
    print("Fixing artifact paths for: ", sys.argv[1])
    fix_artifact_paths(sys.argv[1])
