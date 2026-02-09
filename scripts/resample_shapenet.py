import multiprocessing
import subprocess
import shlex
from shutil import copy2
import os
from glob import glob
import numpy as np
from collections import OrderedDict
from pytorch_points.utils.pc_utils import save_ply
from pytorch_points.misc import logger
from multiprocessing.pool import ThreadPool
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR + "/..")

from common import find_files

def call_proc(cmd):
    """ This runs in a separate thread. """
    try:
        p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        return (out, err)
    except Exception as e:
        return ("", f"Error: {str(e)}")

if __name__ == "__main__":
    # 스레드 수를 CPU 코어 수의 절반으로 설정하여 메모리 사용을 줄임
    N_CORE = max(1, multiprocessing.cpu_count() // 4)  # 코어 수 줄이기
    N_POINT = 20000
    BATCH_SIZE = 500  # 한번에 처리할 파일 수 제한
    RETRY_LIMIT = 3  # 실패한 파일 재시도 횟수

    print("Using %d of %d cores" % (N_CORE, multiprocessing.cpu_count()))

    source_dir = sys.argv[1]  # input directory
    output_dir = sys.argv[2]  # output directory

    SAMPLE_BIN = os.path.join("/usr/local/share/Thea/Build/Output/bin/MeshSample")

    ###################################
    # 1. gather source and target
    ###################################
    source_files = find_files(source_dir, 'obj')
    logger.info("Found {} source files".format(len(source_files)))

    os.makedirs(output_dir, exist_ok=True)

    total_files = len(source_files)
    processed_files = 0

    while processed_files < total_files:
        batch_files = source_files[processed_files:processed_files + BATCH_SIZE]
        results = []

        # 새로운 스레드 풀을 루프 내부에서 생성하여 파일 배치를 처리
        pool = ThreadPool(processes=N_CORE)

        for input_file in batch_files:
            source_name = os.path.splitext(os.path.basename(input_file))[0]
            my_out_dir = os.path.join(output_dir, os.path.relpath(os.path.dirname(input_file), source_dir))
            os.makedirs(my_out_dir, exist_ok=True)
            output_file = os.path.join(my_out_dir, source_name + ".txt")
            # ./MeshSample -n2048 INPUT OUTPUT
            command = f"{SAMPLE_BIN} -n{N_POINT} {input_file} {output_file}"
            results.append(pool.apply_async(call_proc, (command,)))

        # 스레드 풀을 닫고 작업이 완료되기를 기다림
        pool.close()
        pool.join()

        # Error handling and retry logic
        for result in results:
            out, err = result.get()
            if len(err) > 0:
                print("Error during processing: {}".format(err.decode('utf-8')))
            else:
                print("Output: {}".format(out.decode('utf-8')))

        results.clear()
        processed_files += BATCH_SIZE

        # 딜레이를 주어 CPU 과부하 및 메모리 사용량을 줄임
        time.sleep(1)

    print("Processing complete!")
