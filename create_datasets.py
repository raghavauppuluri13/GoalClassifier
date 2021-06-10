import os
import glob
from utility import create_bc_dataset_from_videos 

def main():
    clean_video_paths = [os.path.abspath(path) for path in glob.glob('Data/Clean/clean*') 
    # peel_video_paths = [os.path.abspath(path) for path in glob.glob('Data/Peel/peel*') 
    # write_video_paths = [os.path.abspath(path) for path in glob.glob('Data/Write/write*') 
    test_video_paths = [os.path.abspath(path) for path in glob.glob('Data/Clean/test/clean*') 

    target_dir = 'clean_dataset'

    clean_split_idxs = [165, 104, 176] # determined from finding last case of occlusion or clear task completion

    create_bc_dataset_from_videos(clean_video_paths, clean_split_idxs, target_dir)
    create_bc_dataset_from_videos(test_video_paths, [sys.maxsize], 'test_' + target_dir)
