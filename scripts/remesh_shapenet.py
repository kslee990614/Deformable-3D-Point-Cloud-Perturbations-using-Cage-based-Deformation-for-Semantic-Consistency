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
import pymesh

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR + "/..")

from common import remesh, find_files

def remesh_and_save(input_path, output_path):
    try:
        obj = remesh(input_path)
        pymesh.save_mesh(output_path, obj)
    except Exception as e:
        return ("", str(e))
    else:
        return ("", "")

if __name__ == "__main__":
    N_CORE = 8
    N_POINT = 1024
    print("Using %d of %d cores" % (N_CORE, multiprocessing.cpu_count()))

    source_dir = sys.argv[1]  # 입력 디렉토리
    output_dir = sys.argv[2]  # 출력 디렉토리

    ###################################
    # 1. 소스 및 타겟 파일 수집
    ###################################
    source_files = find_files(source_dir, 'obj')
    logger.info("Found {} source files".format(len(source_files)))

    os.makedirs(output_dir, exist_ok=True)

    ###################################
    # 샘플링 및 저장
    ###################################
    pool = ThreadPool(processes=N_CORE)
    results = []
    for input_file in source_files:
        source_name = os.path.splitext(os.path.basename(input_file))[0]  # 파일명에서 확장자 제거
        # 입력 파일의 디렉토리 경로를 소스 디렉토리를 기준으로 상대 경로로 변환
        relative_dir = os.path.relpath(os.path.dirname(input_file), source_dir)
        # 출력 디렉토리 경로 생성
        my_out_dir = os.path.join(output_dir, relative_dir)
        # 파일명과 동일한 폴더 생성
        model_dir = os.path.join(my_out_dir, source_name)
        os.makedirs(model_dir, exist_ok=True)
        # 출력 파일 경로 설정
        output_file = os.path.join(model_dir, "model.obj")
        results.append(pool.apply_async(remesh_and_save, (input_file, output_file)))

    # 풀을 닫고 각 작업이 완료될 때까지 기다림
    pool.close()
    pool.join()
    for result in results:
        out, err = result.get()
        if len(err) > 0:
            print("err: {}".format(err))
    results.clear()
