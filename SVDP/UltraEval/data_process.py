import os
import shutil
import subprocess


def generate_data(datasets_dir):
    for subdir in os.listdir(datasets_dir):
        subdir_path = os.path.join(datasets_dir, subdir)

        if os.path.isdir(subdir_path):
            script_path = os.path.join(subdir_path, "make_dataset.py")

            if os.path.isfile(script_path):
                try:
                    subprocess.run(["python", script_path], check=True)
                    print(f"Success make_dataset.py in {subdir_path}")
                except:
                    print(f"Fail make_dataset.py in {subdir_path}")
            else:
                print(f"*******No make_dataset.py in {subdir_path}*******")


def del_data(datasets_dir):
    for root, dirs, files in os.walk(datasets_dir):
        if "data" in dirs:
            data_dir_path = os.path.join(root, "data")
            for file in os.listdir(data_dir_path):
                if file.endswith(".jsonl"):
                    shutil.rmtree(data_dir_path)
                    print(f"success delete data folder {data_dir_path}")
                    break


if __name__ == "__main__":
    datasets_dir = "datasets"
    generate_data(datasets_dir)
    # del_data(datasets_dir)
