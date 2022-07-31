import openpifpaf
import torch
import argparse
import copy
import logging
import torch.multiprocessing as mp
import csv
from default_params import *
from helpers import last_ip
import os
import matplotlib.pyplot as plt
import cv2
import time
import receiver
import algorithms
import threading


try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass


class FallDetector:
    def __init__(self, t=DEFAULT_CONSEC_FRAMES):
        self.consecutive_frames = t
        self.args = self.cli()

    def cli(self):
        parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        openpifpaf.network.Factory.cli(parser)
        openpifpaf.decoder.cli(parser)
        parser.add_argument('--resolution', default=0.4, type=float,
                            help=('Resolution prescale factor from 640x480. '
                                  'Will be rounded to multiples of 16.'))
        parser.add_argument('--resize', default=None, type=str,
                            help=('Force input image resize. '
                                  'Example WIDTHxHEIGHT.'))
        parser.add_argument('--num_cams', default=1, type=int,
                            help='Number of Cameras.')
        parser.add_argument('--video', default=None, type=str,
                            help='Path to the video file.\nFor single video fall detection(--num_cams=1), save your videos as abc.xyz and set --video=abc.xyz\nFor 2 video fall detection(--num_cams=2), save your videos as abc1.xyz & abc2.xyz and set --video=abc.xyz')
        parser.add_argument('--debug', default=False, action='store_true',
                            help='debug messages and autoreload')
        parser.add_argument('--disable_cuda', default=False, action='store_true',
                            help='disables cuda support and runs from gpu')

        vis_args = parser.add_argument_group('Visualisation')
        vis_args.add_argument('--plot_graph', default=False, action='store_true',
                              help='Plot the graph of features extracted from keypoints of pose.')
        vis_args.add_argument('--joints', default=True, action='store_true',
                              help='Draw joint\'s keypoints on the output video.')
        vis_args.add_argument('--skeleton', default=True, action='store_true',
                              help='Draw skeleton on the output video.')
        vis_args.add_argument('--coco_points', default=False, action='store_true',
                              help='Visualises the COCO points of the human pose.')
        vis_args.add_argument('--save_output', default=False, action='store_true',
                              help='Save the result in a video file. Output videos are saved in the same directory as input videos with "out" appended at the start of the title')
        vis_args.add_argument('--fps', default=18, type=int,
                              help='FPS for the output video.')
        # vis_args.add_argument('--out-path', default='result.avi', type=str,
        #                       help='Save the output video at the path specified. .avi file format.')

        args = parser.parse_args()

        # Log
        logging.basicConfig(level=logging.INFO if not args.debug else logging.DEBUG)

        args.force_complete_pose = True
        args.instance_threshold = 0.2
        args.seed_threshold = 0.5

        # Add args.device
        args.device = torch.device('cpu')
        args.pin_memory = False
        if not args.disable_cuda and torch.cuda.is_available():
            args.device = torch.device('cuda')
            args.pin_memory = True

        if args.checkpoint is None:
            args.checkpoint = 'shufflenetv2k16-apollo-24'

        openpifpaf.decoder.configure(args)
        openpifpaf.network.Factory.configure(args)

        return args

    def begin(self):
        print('Starting...')
        e = mp.Event()
        queues = [mp.Queue() for _ in range(self.args.num_cams)]
        counter1 = mp.Value('i', 0)
        counter2 = mp.Value('i', 0)
        argss = [copy.deepcopy(self.args) for _ in range(self.args.num_cams)]

        # 改成仅有一个摄像头
        if self.args.video is None:
            argss[0].video = 0
        process1 = threading.Thread(target=algorithms.extract_keypoints_parallel,
                                args=(queues[0], argss[0], counter1, counter2, self.consecutive_frames, e))
        process1.start()
        if self.args.coco_points:
            process1.join()
        else:
            process2 = threading.Thread(target=algorithms.alg2_sequential, args=(queues, argss,
                                                                self.consecutive_frames, e))
            process2.start()

        recvr = receiver.Receiver(draw='amp', subcarrier=20)
        # process3 = threading.Thread(target=recvr.serial_read)
        # process3.start()
        # process4 = threading.Thread(target=recvr.draw)
        # process4.start()
        
        process5 = threading.Thread(target=self.window)
        process5.start()

        process1.join()
        if not self.args.coco_points:
            process2.join()
        # process3.join()
        # process4.join()
        process5.join()

        print('Exiting...')
        return

    def window(self):
        cv2.namedWindow('preview')
        while True:
            camera = algorithms.camera
            print('camera visited')
            if camera is None:
                continue
            # print(algorithms.state)
            # preview = np.concatenate((camera, receiver.image_csi))
            preview = camera
            cv2.imshow('preview', preview)
            algorithms.camera_is_ready = False
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    f = FallDetector()
    f.begin()
