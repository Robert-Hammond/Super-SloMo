import cv2
import os
from tqdm import tqdm

def make_video_from_frames(in_dir, out_file, extension='png', fps=30.0):
    if not os.path.isdir(in_dir):
        print(f"ERROR: Could not find directory \"{in_dir}\"")
        return -1
    
    files = set(os.listdir(in_dir))
    frames = []
    print("Reading frames...")
    for frame_num in tqdm(range(1, len(files) + 1)):
        if f"{frame_num}.png" not in files:
            break
        path = os.path.join(in_dir, f"{frame_num}.{extension}")
        frames.append(cv2.imread(path))
    
    if not frames:
        print(f"ERROR: no PNG files found in directory \"{in_dir}\"")
        return -1
    
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    writer = cv2.VideoWriter(out_file, fourcc, fps, (width, height))
    print("Creating video...")
    for i in tqdm(range(len(frames))):
        writer.write(frames[i])
    writer.release()
    return 0


def make_frames_from_video(in_file, out_dir, extension='png'):
    if not os.path.isfile(in_file):
        print(f"ERROR: Could not find file \"{in_file}\"")
        return -1
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    reader = cv2.VideoCapture(in_file)

    frame_num = 1
    print("Extracting frames...")
    while True:
        rc, im = reader.read()
        if not rc:
            break
        cv2.imwrite(os.path.join(out_dir, f"{frame_num:06d}.{extension}"), im)
        frame_num += 1
        
    return frame_num
    
        