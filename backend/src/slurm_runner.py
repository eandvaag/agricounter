
import logging
import os
import sys
import argparse
import shutil
import uuid
import time
import datetime
import subprocess


from io_utils import json_io

from models.yolov4 import driver as yolov4_driver




def write_slurm_job(out_path, slurm_job_config):

    global_slurm_config_path = os.path.join("slurm_config.json")
    global_slurm_config = json_io.load_json(global_slurm_config_path)


    f = open(out_path, "w")
    f.write(
        "#!/bin/bash\n" +
        "\n" +
        "#SBATCH --job-name=" + slurm_job_config["job_name"] + "\n" +
        "#SBATCH --output=" + slurm_job_config["job_out_path"] + "\n" +
        "#SBATCH --account=" + global_slurm_config["account"] + "\n" +
        "#SBATCH --time=" + slurm_job_config["time_estimate"] + "\n" +
        "#SBATCH --ntasks=1\n" +
        "#SBATCH --mem=" + slurm_job_config["memory_estimate"] + "\n" +
        "#SBATCH --gpus-per-node=1\n"
    )
    

    for opt_key in ["gres", "partition", "cluster"]:
        if opt_key in global_slurm_config:
            f.write("#SBATCH --" + opt_key + "=" + global_slurm_config[opt_key] + "\n")


    f.write(
        "\n" +
        "apptainer run --nv ac_hpc.sif " + slurm_job_config["slurm_dir"] + "\n"
    )

    f.close()




def create_and_run_slurm_job(job, slurm_dir, time_estimate):

    logger = logging.getLogger(__name__)

    if os.path.exists(slurm_dir):
        shutil.rmtree(slurm_dir)
    os.makedirs(slurm_dir)


    slurm_job_path = os.path.join(slurm_dir, "job.sh")
    slurm_job_out_path = os.path.join(slurm_dir, "job.out")
    slurm_job_params_path = os.path.join(slurm_dir, "job_params.json")


    json_io.save_json(slurm_job_params_path, job)


    slurm_job_config = {
        "job_name": "agricounter_" + str(uuid.uuid4()),
        "job_out_path": slurm_job_out_path,
        "time_estimate": time_estimate,
        "memory_estimate": "10GB",
        "slurm_dir": slurm_dir
    }

    write_slurm_job(slurm_job_path, slurm_job_config)


    #test_cmd = "apptainer run --nv ac_hpc.sif " + slurm_job_config["slurm_dir"]
    return
    slurm_cmd = "sbatch " + slurm_job_path
    start_time = time.time()
    subprocess.run(slurm_cmd.split(" "))
    end_time = time.time()
    elapsed = str(datetime.timedelta(seconds=round(end_time - start_time)))
    logger.info("Slurm job completed. Elapsed time: {}".format(elapsed))


    slurm_job_success_path = os.path.join(slurm_dir, "completed_successfully.txt")
    if not os.path.exists(slurm_job_success_path):
        raise RuntimeError("Slurm job error.")




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