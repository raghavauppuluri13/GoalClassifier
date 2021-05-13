import os
import pathlib
import cv2

import matplotlib as plt

def create_bc_dataset_from_videos(video_paths, split_idxs, target_dir):
    '''
    Creates binary classifier dataset from demonstration videos and indicies
    representing split between positive and negative examples

    ARGS:
    video_paths (list(str)): absolute path to videos to extract
    split_idxs (list(int)): index to split into negative and positive frames respectively, index is a positive example
    target_dir (str): target directory to write frames
    '''

    success_cnt = 0
    fail_cnt = 0

    success_dir = target_dir + '/success'
    fail_dir = target_dir + '/fail'

    pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(success_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(fail_dir).mkdir(parents=True, exist_ok=True)

    for i, video_path in enumerate(video_paths):
        success = True
        vidcap = cv2.VideoCapture(video_path)
        while success:
            success, image = vidcap.read()
            if(success):
                is_success = bool(success_cnt + fail_cnt >= split_idxs[i])
                if(is_success):
                    file_name = str(success_cnt) + ".jpg"
                    path = os.path.join(success_dir, file_name)
                    cv2.imwrite(path, image)
                    success_cnt = success_cnt + 1
                else:
                    file_name = str(fail_cnt) + ".jpg"
                    path = os.path.join(fail_dir, file_name)
                    cv2.imwrite(path, image)
                    fail_cnt = fail_cnt + 1
            else:
                print("Error reading frame: ", success_cnt + fail_cnt)
            if vidcap.get(cv2.CAP_PROP_POS_FRAMES) == vidcap.get(cv2.CAP_PROP_FRAME_COUNT):
                break
