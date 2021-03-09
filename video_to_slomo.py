#!/usr/bin/env python3
import argparse
import os
import os.path
import ctypes
from shutil import rmtree, move
from PIL import Image
import torch
from torch._C import set_flush_denormal
import torchvision.transforms as transforms
import model
import dataloader
import platform
from tqdm import tqdm
import time
import warnings
import frametools
warnings.filterwarnings("ignore")

# For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--ffmpeg_dir", type=str, default="", help='path to ffmpeg.exe')
parser.add_argument("--video", type=str, required=True, help='path of video to be converted')
parser.add_argument("--checkpoint", type=str, required=True, help='path of checkpoint for pretrained model')
parser.add_argument("--fps", type=float, default=30, help='specify fps of output video. Default: 30.')
parser.add_argument("--sf", type=int, required=True, help='specify the slomo factor N. This will increase the frames by Nx. Example sf=2 ==> 2x frames')
parser.add_argument("--batch_size", type=int, default=1, help='Specify batch size for faster conversion. This will depend on your cpu/gpu memory. Default: 1')
parser.add_argument("--output", type=str, default=".mkv", help='Specify output file name.')
args = parser.parse_args()

def check():
    """
    Checks the validity of commandline arguments.

    Parameters
    ----------
        None

    Returns
    -------
        error : string
            Error message if error occurs otherwise blank string.
    """


    error = ""
    if (args.sf < 2):
        error = "Error: --sf/slomo factor has to be atleast 2"
    if (args.batch_size < 1):
        error = "Error: --batch_size has to be atleast 1"
    if (args.fps < 1):
        error = "Error: --fps has to be atleast 1"
    if ".mkv" not in args.output:
        error = "output needs to have mkv container"
    return error


def create_video(dir):
    IS_WINDOWS = 'Windows' == platform.system()

    if IS_WINDOWS:
        ffmpeg_path = os.path.join(args.ffmpeg_dir, "ffmpeg")
    else:
        ffmpeg_path = "ffmpeg"

    error = ""
    dot = args.video.find('.', len(args.video) - 5)
    vid_name = args.video[:dot]
    # extension = args.video[dot:]
    out_file = f'{vid_name}({args.sf}x).mkv'
    print('{} -r {} -i {}/%d.png -vcodec ffvhuff {}'.format(ffmpeg_path, args.fps, dir, out_file))
    retn = os.system('{} -r {} -i {}/%d.png -vcodec ffvhuff "{}"'.format(ffmpeg_path, args.fps, dir, out_file))
    if retn:
        error = "Error creating output video. Exiting."
    return error

def interpolate_frames(extractionPath, outputPath):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mean = [0.429, 0.431, 0.397]
    std  = [1, 1, 1]
    normalize = transforms.Normalize(mean=mean, std=std)

    negmean = [x * -1 for x in mean]
    revNormalize = transforms.Normalize(mean=negmean, std=std)

    # Temporary fix for issue #7 https://github.com/avinashpaliwal/Super-SloMo/issues/7 -
    # - Removed per channel mean subtraction for CPU.
    if (device == "cpu"):
        transform = transforms.Compose([transforms.ToTensor()])
        TP = transforms.Compose([transforms.ToPILImage()])
    else:
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        TP = transforms.Compose([revNormalize, transforms.ToPILImage()])

    # Load data
    videoFrames = dataloader.Video(root=extractionPath, transform=transform)
    videoFramesloader = torch.utils.data.DataLoader(videoFrames, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    flowComp = model.UNet(6, 4)
    flowComp.to(device)
    for param in flowComp.parameters():
        param.requires_grad = False
    ArbTimeFlowIntrp = model.UNet(20, 5)
    ArbTimeFlowIntrp.to(device)
    for param in ArbTimeFlowIntrp.parameters():
        param.requires_grad = False

    flowBackWarp = model.backWarp(videoFrames.dim[0], videoFrames.dim[1], device)
    flowBackWarp = flowBackWarp.to(device)

    dict1 = torch.load(args.checkpoint, map_location='cpu')
    ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
    flowComp.load_state_dict(dict1['state_dictFC'])

    print("Interpolating intermediate frames...")
    frameCounter = 1
    with torch.no_grad():
        for _, (frame0, frame1) in enumerate(tqdm(videoFramesloader), 0):
            fail = True
            while fail:
                try:
                    # print(f"frameCounter = {frameCounter}")
                    if os.path.isfile(os.path.join(outputPath, str(frameCounter + args.sf*args.batch_size - 1) + ".png")):
                        frameCounter += args.sf * args.batch_size
                        continue
                    I0 = frame0.to(device)
                    I1 = frame1.to(device)

                    flowOut = flowComp(torch.cat((I0, I1), dim=1))
                    F_0_1 = flowOut[:,:2,:,:]
                    F_1_0 = flowOut[:,2:,:,:]

                    # Save reference frames in output folder
                    for batchIndex in range(args.batch_size):
                        (TP(frame0[batchIndex].detach())).resize(videoFrames.origDim, Image.BILINEAR) \
                            .save(os.path.join(outputPath, str(frameCounter + args.sf * batchIndex) + ".png"))
                    frameCounter += 1

                    # Generate intermediate frames
                    for intermediateIndex in range(1, args.sf):
                        # print(f"\tframeCounter = {frameCounter}")
                        save_files = [os.path.join(outputPath, str(frameCounter + args.sf * batchIndex) + ".png")
                            for batchIndex in range(args.batch_size)]
                        
                        t = float(intermediateIndex) / args.sf
                        temp = -t * (1 - t)
                        fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

                        F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                        F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

                        g_I0_F_t_0 = flowBackWarp(I0, F_t_0)
                        g_I1_F_t_1 = flowBackWarp(I1, F_t_1)

                        intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

                        F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                        F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                        V_t_0   = torch.sigmoid(intrpOut[:, 4:5, :, :])
                        V_t_1   = 1 - V_t_0

                        g_I0_F_t_0_f = flowBackWarp(I0, F_t_0_f)
                        g_I1_F_t_1_f = flowBackWarp(I1, F_t_1_f)

                        wCoeff = [1 - t, t]

                        Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

                        # Save intermediate frame
                        for batchIndex in range(args.batch_size):
                            (TP(Ft_p[batchIndex].cpu().detach())).resize(videoFrames.origDim, Image.BILINEAR).save(save_files[batchIndex])
                        frameCounter += 1
                        # gc.collect()
                        torch.cuda.empty_cache() # to avoid GPU out of memory error
                        # torch.cuda.synchronize()
                    # Set counter accounting for batching of frames
                    frameCounter += args.sf * (args.batch_size - 1)
                    fail = False
                except RuntimeError:
                    time.sleep(5) # give it some time to literally cool down


def main():
    # Check if arguments are okay
    error = check()
    if error:
        print(error)
        exit(1)

    # Create extraction folder and extract frames
    IS_WINDOWS = 'Windows' == platform.system()
    extractionDir = "tmpSuperSloMo"
    if not IS_WINDOWS:
        # Assuming UNIX-like system where "." indicates hidden directories
        extractionDir = "." + extractionDir
    # COMMENT THESE IF ALREADY EXTRACTED FRAMES
    # -------------------------------------------
    if os.path.isdir(extractionDir):
        rmtree(extractionDir)
    os.mkdir(extractionDir)
    # -------------------------------------------
    if IS_WINDOWS:
        FILE_ATTRIBUTE_HIDDEN = 0x02
        # ctypes.windll only exists on Windows
        ctypes.windll.kernel32.SetFileAttributesW(extractionDir, FILE_ATTRIBUTE_HIDDEN)

    extractionPath = os.path.join(extractionDir, "input")
    outputPath     = os.path.join(extractionDir, "output")
    # COMMENT THESE IF ALREADY EXTRACTED FRAMES
    # -------------------------------------------
    os.mkdir(extractionPath)
    os.mkdir(outputPath)
    initial_frame_count = frametools.make_frames_from_video(args.video, extractionPath, extension='png')
    if initial_frame_count == -1:
        print(error)
        exit(1)
    # -------------------------------------------

    interpolate_frames(extractionPath, outputPath)

    # Generate video from interpolated frames
    dot = args.video.find('.', len(args.video) - 5)
    vid_name = args.video[:dot]
    ext = args.video[dot:]
    out_file = f'{vid_name}({args.sf}x){ext}'
    rc = frametools.make_video_from_frames(outputPath, out_file, extension='png', fps=args.fps)

    if rc != -1:
        # Remove temporary files upon successful conversion
        rmtree(extractionDir)
        exit(0)

main()

