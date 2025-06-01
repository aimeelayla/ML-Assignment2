import os
import shutil
import subprocess
import sys

import pandas as pd


BASE_PATH = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))

# Join and normalise path
def join_paths(*args):
    return os.path.normpath(os.path.join(*args))


def main():

    # Delete
    if os.path.exists(join_paths(BASE_PATH, "predlabels.txt")):
        os.remove(join_paths(BASE_PATH, "predlabels.txt"))
    if os.path.exists(join_paths(BASE_PATH, "traindata.txt")):
        os.remove(join_paths(BASE_PATH, "traindata.txt"))
    if os.path.exists(join_paths(BASE_PATH, "trainlabels.txt")):
        os.remove(join_paths(BASE_PATH, "trainlabels.txt"))

    # Run classifyall.py
    try:
        python_cmd = os.path.basename(sys.executable).replace(".exe", "")
        # print()
        cmd_str = f"{python_cmd} classifyall.py"
        os.system(cmd_str)
        # result = subprocess.run(cmd_str, check=True, text=True)

        # print(f"classifyall.py ran without errors and gave exit code: {result.returncode}")
    except Exception as e:
        raise Exception(f"classifyall.py failed with error: {e}")


    # Validate predlabels.txt
    test_data_df = pd.read_csv("testdata.txt", header=None)
    pred_path = "predlabels.txt"
    if not os.path.exists(pred_path):
        raise FileNotFoundError("predlabels.txt was not created or not saved in same folder as classifyall.py.")
    pred_labels_df = pd.read_csv(pred_path, header=None)
    if pred_labels_df.shape != (test_data_df.shape[0], 1):
        raise ValueError(f"predlabels.txt is of wrong shape. It should be {(test_data_df.shape[0], 1)} but is {pred_labels_df.shape}")


    if not pred_labels_df.iloc[:, 0].between(0, 9).all():
        raise ValueError("Invalid values in predlabels.txt. All values should be between 0 and 9.")

    print("#####################")
    print("# VALIDATION PASSED #")
    print("#####################")

    if os.path.exists(join_paths(BASE_PATH, "predlabels.txt")):
        os.remove(join_paths(BASE_PATH, "predlabels.txt"))


if __name__ == "__main__":
    main()