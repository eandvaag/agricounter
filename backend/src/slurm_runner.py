
import os
import sys
import argparse

from io_utils import json_io

from models.yolov4 import driver as yolov4_driver




def run(slurm_dir):

    slurm_job_params_path = os.path.join(slurm_dir, "job_params.json")
    job = json_io.load_json(slurm_job_params_path)

    task = job["task"]

    if task == "predict":
        yolov4_driver.predict(job)

    elif task == "fine_tune":
        yolov4_driver.fine_tune(job)

    slurm_job_success_path = os.path.join(slurm_dir, "completed_successfully.txt")
    open(slurm_job_success_path, "w").close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="slurm_runner",
        description="Program for running SLURM jobs"
    )

    parser.add_argument("slurm_dir", type=str)

    
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        exit(1)

    if len(sys.argv) > 2:
        parser.print_help()
        exit(1)

    run(args.slurm_dir)