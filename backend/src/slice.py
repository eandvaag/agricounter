import logging
import os
import glob
import argparse
import subprocess
import time
import traceback
# import concurrent.futures
import threading

from io_utils import json_io
from lock_queue import LockQueue


# accepted_extensions = [".jpg", ".JPG", ".png", ".PNG"]
accepted_extensions = ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]
NUM_WORKERS = 10

image_queue = LockQueue()
operation_error = False
image_set_dir = None
is_ortho = False

def thread_function(index):
    global operation_error

    logger = logging.getLogger(__name__)

    # print("Thread {} starting".format(index))
    images_dir = os.path.join(image_set_dir, "images")
    dzi_images_dir = os.path.join(image_set_dir, "dzi_images")
    image_queue_size = image_queue.size()
    while image_queue_size > 0:
        image_path = image_queue.dequeue()
        image_name = os.path.basename(image_path)
        # print("worker thread dequeued", image_path)
        split_image_name = image_name.split(".")
        dzi_path = os.path.join(dzi_images_dir, split_image_name[0])
        image_extension = split_image_name[-1]
        # print("dzi_path", dzi_path)

        if (image_extension not in accepted_extensions) and not is_ortho:
            # print("running convert")
            conv_path = os.path.join(images_dir, split_image_name[0] + ".png")
            try:
                subprocess.run(["convert", image_path, conv_path], check=True)
            except Exception as e:
                trace = traceback.format_exc()
                logger.error("Error from thread {}: {}".format(index, trace))
                operation_error = True
        else:
            # print("not running convert")
            conv_path = image_path
        
        # print("checking for error", operation_error)
        if operation_error:
            logger.info("Thread {}: Error detected, returning.".format(index))
            return

        conv_path = image_path

        # print("running magick slicer")
        # start_time = time.time()
        try:
            # subprocess.run(["./MagickSlicer/magick-slicer.sh", "-v1", conv_path, dzi_path], check=True)
            subprocess.run(["vips", "dzsave", conv_path, dzi_path])
        except Exception as e:
            trace = traceback.format_exc()
            logger.error("Error from thread {}: {}".format(index, trace))
            operation_error = True
        
        
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print("Finished dzi conversion. Elapsed time: {}".format(elapsed_time))

        if operation_error:
            logger.info("Thread {}: Error detected, returning.".format(index))
            return

        image_queue_size = image_queue.size()


    # print("worker {} has nothing to do, returning now".format(index))
    return




def start():

    logger = logging.getLogger(__name__)

    metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
    metadata = json_io.load_json(metadata_path)
    global is_ortho
    is_ortho = metadata["is_ortho"] == "yes"

    image_paths = glob.glob(os.path.join(image_set_dir, "images", "*"))


    # print("image_paths", image_paths)

    for image_path in image_paths:
        image_queue.enqueue(image_path)
    
    logger.info("Starting worker threads for DZI image conversion.")
    start_time = time.time()

    threads = []
    for i in range(NUM_WORKERS):
        x = threading.Thread(target=thread_function, args=(i,))
        threads.append(x)

    for x in threads:
        x.start()

    for x in threads:
        x.join()

    end_time = time.time()
    elapsed_time = end_time - start_time

    # with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    #     executor.map(thread_function, range(MAX_WORKERS))

    # print("Main thread: exiting")
    if operation_error:
        logger.info("Exiting DZI image conversion with error.")
        exit(1)
    else:
        logger.info("Exiting DZI image conversion. Conversion was successful and took {} seconds.".format(elapsed_time))
        exit(0)





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("image_set_dir", type=str)
    
    args = parser.parse_args()
    image_set_dir = args.image_set_dir


    start()

    