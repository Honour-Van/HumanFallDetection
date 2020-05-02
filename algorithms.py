import cv2
import csv
import logging
import base64
import time
import numpy as np
import matplotlib.pyplot as plt
from visual import write_on_image, visualise
from processor import Processor
from helpers import pop_and_add, last_ip, dist
from default_params import *
from writer import write_to_csv
from inv_pendulum import *


def alg2(queue, plot_graph, consecutive_frames=DEFAULT_CONSEC_FRAMES):
    re_matrix = []
    max_length_mat = 101
    if not plot_graph:
        max_length_mat = consecutive_frames
    ip_set = []

    while True:
        if queue.qsize() > 0:
            new_frame = queue.get()
            if new_frame is None:
                break
            match_ip(ip_set, new_frame, re_matrix, max_length_mat)
            for i in range(len(ip_set)):
                if ip_set[i][-1] is not None:
                    if ip_set[i][-2] is not None:
                        pop_and_add(re_matrix[i], get_rot_energy(
                                    ip_set[i][-2], ip_set[i][-1]), max_length_mat - 1)
                    elif ip_set[i][-3] is not None:
                        pop_and_add(re_matrix[i], get_rot_energy(
                                    ip_set[i][-3], ip_set[i][-1]), max_length_mat - 1)
                    elif ip_set[i][-4] is not None:
                        pop_and_add(re_matrix[i], get_rot_energy(
                                    ip_set[i][-4], ip_set[i][-1]), max_length_mat - 1)
                    else:
                        pop_and_add(re_matrix[i], 0, max_length_mat)
                else:
                    pop_and_add(re_matrix[i], 0, max_length_mat)

        if len(re_matrix) > 0:
            plt.clf()
            x = np.linspace(1, len(re_matrix[0]), len(re_matrix[0]))
            axes = plt.gca()
            line, = axes.plot(x, re_matrix[0], 'r-')
            plt.draw()
            plt.pause(1e-17)

    # plt.show()


def match_ip(ip_set, new_ips, re_matrix, consecutive_frames=DEFAULT_CONSEC_FRAMES):
    len_ip_set = len(ip_set)
    added = [False for _ in range(len_ip_set)]
    for new_ip in new_ips:
        if not is_valid(new_ip):
            continue
        cmin = [MIN_THRESH, -1]
        for i in range(len_ip_set):
            print(dist(last_ip(ip_set[i]), new_ip))
            if not added[i] and dist(last_ip(ip_set[i]), new_ip) < cmin[0]:
                cmin[0] = dist(last_ip(ip_set[i]), new_ip)
                cmin[1] = i

        if cmin[1] == -1:
            ip_set.append([None for _ in range(consecutive_frames - 1)] + [new_ip])
            re_matrix.append([])
        else:
            added[cmin[1]] = True
            pop_and_add(ip_set[cmin[1]], new_ip, consecutive_frames)

    i = 0
    while i < len(added):
        if not added[i]:
            pop_and_add(ip_set[i], None, consecutive_frames)
            if ip_set[i] == [None for _ in range(consecutive_frames)]:
                ip_set.pop(i)
                re_matrix.pop(i)
                added.pop(i)
                continue
        i += 1


def extract_keypoints(queue, args, consecutive_frames=DEFAULT_CONSEC_FRAMES):
    print('main started')
    # M = []
    if args.video is None:
        logging.debug('Video source: webcam')
        cam = cv2.VideoCapture(0)
    else:
        logging.debug(f'Video source: {args.video}')
        cam = cv2.VideoCapture(args.video)

    # Setup CSV file
    with open(args.csv_path, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['frame_no',
                             'nose.x', 'nose.y', 'nose.prob',
                             'l.eye.x', 'l.eye.y', 'l.eye.prob',
                             'r.eye.x', 'r.eye.y', 'r.eye.prob',
                             'l.ear.x', 'l.ear.y', 'l.ear.prob',
                             'r.ear.x', 'r.ear.y', 'r.ear.prob',
                             'l.shoulder.x', 'l.shoulder.y', 'l.shoulder.prob',
                             'r.shoulder.x', 'r.shoulder.y', 'r.shoulder.prob',
                             'l.elbow.x', 'l.elbow.y', 'l.elbow.prob',
                             'r.elbow.x', 'r.elbow.y', 'r.elbow.prob',
                             'l.wrist.x', 'l.wrist.y', 'l.wrist.prob',
                             'r.wrist.x', 'r.wrist.y', 'r.wrist.prob',
                             'l.hip.x', 'l.hip.y', 'l.hip.prob',
                             'r.hip.x', 'r.hip.y', 'r.hip.prob',
                             'l.knee.x', 'l.knee.y', 'l.knee.prob',
                             'r.knee.x', 'r.knee.y', 'r.knee.prob',
                             'l.ankle.x', 'l.ankle.y', 'l.ankle.prob',
                             'r.ankle.x', 'r.ankle.y', 'r.ankle.prob',
                             ])

    ret_val, img = cam.read()

    # Resize the video
    if args.resize is None:
        height, width = img.shape[:2]
    else:
        width, height = [int(dim) for dim in args.resize.split('x')]

    # width_height = (width, height)
    width_height = (int(width * args.resolution // 16) * 16,
                    int(height * args.resolution // 16) * 16)
    logging.debug(f'Target width and height = {width_height}')
    processor_singleton = Processor(width_height, args)

    # output_video = None

    frame = 0
    fps = 0
    t0 = time.time()
    cv2.namedWindow('Detected Pose')

    while True:

        ret_val, img = cam.read()
        if img is None:
            print('no more images captured')
            queue.put(None)
            break

        if cv2.waitKey(1) == 27 or cv2.getWindowProperty('Detected Pose', cv2.WND_PROP_VISIBLE) < 1:
            queue.put(None)
            break

        img = cv2.resize(img, (width, height))
        frame += 1

        keypoint_sets, scores, width_height = processor_singleton.single_image(
            b64image=base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('UTF-8')
        )

        if args.coco_points:
            keypoint_sets = [keypoints.tolist() for keypoints in keypoint_sets]
            # keypoint_sets=[{
            #     'coordinates': keypoints.tolist(),
            #     'detection_id': i,
            #     'score': score,
            #     'width_height': width_height,
            # } for i, (keypoints, score) in enumerate(zip(keypoint_sets, scores))]
        else:
            keypoint_sets = [get_kp(keypoints.tolist()) for keypoints in keypoint_sets]
            # keypoint_sets=[{
            #     'coordinates': get_kp(keypoints.tolist()),
            #     'detection_id': i,
            #     'score': score,
            #     'width_height': width_height,
            # } for i, (keypoints, score) in enumerate(zip(keypoint_sets, scores))]

        queue.put(keypoint_sets)

        img = visualise(img=img, keypoint_sets=keypoint_sets, width=width, height=height, vis_keypoints=args.joints,
                        vis_skeleton=args.skeleton, CocoPointsOn=args.coco_points)

        # write_to_csv(frame_number=frame, humans=keypoint_sets,
        #              width=width, height=height, csv_fp=args.csv_path)

        img = write_on_image(
            img=img, text=f"Avg FPS {frame//(time.time()-t0)}", color=[0, 0, 0])

        # print(f"...............{cv2.getWindowProperty('Detected Pose', 1)}................")

        cv2.imshow('Detected Pose', img)

    queue.put(None)
