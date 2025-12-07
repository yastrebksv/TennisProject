import cv2
from court_detection_net import CourtDetectorNet
import numpy as np
from court_reference import CourtReference
from bounce_detector import BounceDetector
from person_detector import PersonDetector
from ball_detector import BallDetector
from utils import scene_detect
import argparse
import torch
from scipy.signal import find_peaks
from scipy.signal import argrelextrema

import csv
import os


def get_projected_coords(ball_track, persons_top, persons_bottom, homography_matrices):
    """
    Transforms Image Coordinates (pixels) -> Minimap Coordinates (meters)
    using the homography matrix.
    """
    proj_ball = []
    proj_top = []
    proj_bot = []

    for i in range(len(homography_matrices)):
        matrix = homography_matrices[i]

        # --- Helper to transform a single point ---
        def transform(x, y):
            if x is None or y is None or matrix is None:
                return None, None

            # Reshape point for cv2.perspectiveTransform
            point = np.array([[[x, y]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(point, matrix)[0][0]
            return transformed[0], transformed[1]

        # 1. Project Ball
        bx, by = ball_track[i] if ball_track[i] else (None, None)
        mb_x, mb_y = transform(bx, by)

        # Optional: Clamp negative Y values like the video drawing does
        if mb_y is not None and mb_y < 0:
            mb_y = 0

        proj_ball.append((mb_x, mb_y))

        # 2. Project Top Player
        # Extract center (x,y) from persons_top list structure
        if persons_top[i] and len(persons_top[i][0]) > 0:
            # persons_top[i][0][1] is the (x,y) center position
            tx, ty = persons_top[i][0][1]
            mt_x, mt_y = transform(tx, ty)
            proj_top.append((mt_x, mt_y))
        else:
            proj_top.append((None, None))

        # 3. Project Bottom Player
        if persons_bottom[i] and len(persons_bottom[i][0]) > 0:
            bx_p, by_p = persons_bottom[i][0][1]
            mbot_x, mbot_y = transform(bx_p, by_p)
            proj_bot.append((mbot_x, mbot_y))
        else:
            proj_bot.append((None, None))

    return proj_ball, proj_top, proj_bot


def write_csv(path_csv, frames, ball_track, bounces, hits, persons_top, persons_bottom,
              homography_matrices, kps_court,
              proj_ball, proj_top, proj_bot):  # <--- Added new arguments

    with open(path_csv, mode='w', newline='') as file:
        writer = csv.writer(file)

        header = ['frame_id',
                  'ball_x', 'ball_y', 'bounce', 'hit',
                  'top_player_x', 'top_player_y',
                  'bottom_player_x', 'bottom_player_y',
                  'top_player_bbox_x1', 'top_player_bbox_y1', 'top_player_bbox_x2', 'top_player_bbox_y2',
                  'bottom_player_bbox_x1', 'bottom_player_bbox_y1', 'bottom_player_bbox_x2', 'bottom_player_bbox_y2',
                  'homography_valid',
                  # --- NEW COLUMNS ---
                  'minimap_ball_x', 'minimap_ball_y',
                  'minimap_top_x', 'minimap_top_y',
                  'minimap_bot_x', 'minimap_bot_y'] + \
                 [f'court_kp_{i}_x' for i in range(14)] + [f'court_kp_{i}_y' for i in range(14)]

        writer.writerow(header)

        for i in range(len(frames)):
            # ... existing extraction logic ...
            ball_x, ball_y = ball_track[i] if ball_track[i] else (None, None)
            bounce = i in bounces
            hit = i in hits

            def extract_person_info(person_list):
                if person_list and len(person_list[0]) > 0:
                    bbox = person_list[0][0]
                    pos = person_list[0][1]
                    return pos[0], pos[1], bbox[0], bbox[1], bbox[2], bbox[3]
                return None, None, None, None, None, None

            top_x, top_y, top_x1, top_y1, top_x2, top_y2 = extract_person_info(persons_top[i])
            bot_x, bot_y, bot_x1, bot_y1, bot_x2, bot_y2 = extract_person_info(persons_bottom[i])

            homography_valid = homography_matrices[i] is not None

            kps = kps_court[i]
            if kps is not None:
                kps_x = [kp[0, 0] for kp in kps]
                kps_y = [kp[0, 1] for kp in kps]
            else:
                kps_x = [None] * 14
                kps_y = [None] * 14

            # --- NEW DATA ---
            mb_x, mb_y = proj_ball[i]
            mt_x, mt_y = proj_top[i]
            mbot_x, mbot_y = proj_bot[i]

            row = [
                      i,
                      ball_x, ball_y, bounce, hit,
                      top_x, top_y,
                      bot_x, bot_y,
                      top_x1, top_y1, top_x2, top_y2,
                      bot_x1, bot_y1, bot_x2, bot_y2,
                      homography_valid,
                      mb_x, mb_y,  # Minimap Ball
                      mt_x, mt_y,  # Minimap Top Player
                      mbot_x, mbot_y  # Minimap Bot Player
                  ] + kps_x + kps_y

            writer.writerow(row)


def read_video(path_video):
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break    
    cap.release()
    return frames, fps

def get_court_img():
    court_reference = CourtReference()
    court = court_reference.build_court_reference()
    court = cv2.dilate(court, np.ones((10, 10), dtype=np.uint8))
    court_img = (np.stack((court, court, court), axis=2)*255).astype(np.uint8)
    return court_img

def main(frames, scenes, bounces, hits, ball_track, homography_matrices, kps_court, persons_top, persons_bottom,
         draw_trace=False, trace=7):
    """
    :params
        frames: list of original images
        scenes: list of beginning and ending of video fragment
        bounces: list of image numbers where ball touches the ground
        ball_track: list of (x,y) ball coordinates
        homography_matrices: list of homography matrices
        kps_court: list of 14 key points of tennis court
        persons_top: list of person bboxes located in the top of tennis court
        persons_bottom: list of person bboxes located in the bottom of tennis court
        draw_trace: whether to draw ball trace
        trace: the length of ball trace
    :return
        imgs_res: list of resulting images
    """
    imgs_res = []
    width_minimap = 166
    height_minimap = 350

    # h_court, w_court = get_court_img().shape[:2]
    # frame_h, frame_w, _ = frames[0].shape
    #
    # # scale minimap to ~20% of frame width
    # width_minimap = int(frame_w * 0.2)
    # height_minimap = int(h_court * (width_minimap / w_court))
    #
    is_track = [x is not None for x in homography_matrices]

    ball_positions = []


    for num_scene in range(len(scenes)):
        sum_track = sum(is_track[scenes[num_scene][0]:scenes[num_scene][1]])
        len_track = scenes[num_scene][1] - scenes[num_scene][0]

        eps = 1e-15
        scene_rate = sum_track/(len_track+eps)
        if (scene_rate > 0.5):
            court_img = get_court_img()
            print("Court: ", court_img)

            for i in range(scenes[num_scene][0], scenes[num_scene][1]):



                img_res = frames[i]
                inv_mat = homography_matrices[i]

                # draw ball trajectory
                if ball_track[i][0]:
                    if draw_trace:
                        for j in range(0, trace):
                            if i-j >= 0:
                                if ball_track[i-j][0]:
                                    draw_x = int(ball_track[i-j][0])
                                    draw_y = int(ball_track[i-j][1])
                                    img_res = cv2.circle(frames[i], (draw_x, draw_y),
                                    radius=3, color=(0, 255, 0), thickness=2)
                    else:    
                        img_res = cv2.circle(img_res , (int(ball_track[i][0]), int(ball_track[i][1])), radius=5,
                                             color=(0, 255, 0), thickness=2)
                        img_res = cv2.putText(img_res, 'ball', 
                              org=(int(ball_track[i][0]) + 8, int(ball_track[i][1]) + 8),
                              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                              fontScale=0.8,
                              thickness=2,
                              color=(0, 255, 0))

                # draw court keypoints
                if kps_court[i] is not None:
                    for j in range(len(kps_court[i])):
                        img_res = cv2.circle(img_res, (int(kps_court[i][j][0, 0]), int(kps_court[i][j][0, 1])),
                                          radius=0, color=(0, 0, 255), thickness=10)

                height, width, _ = img_res.shape


                # ball_point = ball_track[i]
                # ball_point = np.array(ball_point, dtype=np.float32).reshape(1, 1, 2)
                # ball_point = cv2.perspectiveTransform(ball_point, inv_mat)
                # if ball_point[0, 0, 1] < 0:
                #     ball_point[0, 0, 1] = 0
                # court_img = cv2.circle(court_img, (int(ball_point[0, 0, 0]), int(ball_point[0, 0, 1])),
                #                        radius=0, color=(0, 255, 255), thickness=40)

                #
                # ball_point = ball_track[i]
                # ball_point = np.array(ball_point, dtype=np.float32).reshape(1, 1, 2)
                # ball_point = cv2.perspectiveTransform(ball_point, inv_mat)
                # court_img = cv2.circle(court_img, (int(new_ball_positions[i][0]), int(new_ball_positions[i][1])),
                #                        radius=0, color=(0, 255, 255), thickness=20)

                # draw bounce in minimap
                if i in bounces and inv_mat is not None:
                    print(f"❌ Bounce detected at frame {i}")
                    ball_point = ball_track[i]
                    ball_point = np.array(ball_point, dtype=np.float32).reshape(1, 1, 2)
                    ball_point = cv2.perspectiveTransform(ball_point, inv_mat)
                    if ball_point[0, 0, 1] < 0:
                        ball_point[0, 0, 1] = 0
                    court_img = cv2.circle(court_img, (int(ball_point[0, 0, 0]), int(ball_point[0, 0, 1])),
                                           radius=0, color=(0, 255, 255), thickness=50)
                    print(f"❌ Bounce detected at frame {i}", ball_point)




                # draw hits in minimap
                if i in hits and inv_mat is not None:
                    print(f"❌ Bounce detected at frame {i}")
                    ball_point = ball_track[i]
                    ball_point = np.array(ball_point, dtype=np.float32).reshape(1, 1, 2)
                    ball_point = cv2.perspectiveTransform(ball_point, inv_mat)
                    if ball_point[0, 0, 1] < 0 :
                        ball_point[0, 0, 1] = 0
                    court_img = cv2.circle(court_img, (int(ball_point[0, 0, 0]), int(ball_point[0, 0, 1])),
                                                       radius=0, color=(0, 128, 255), thickness=50)
                    print(f"❌ hits detected at frame {i}", ball_point)

                minimap = court_img.copy()

                # draw persons
                persons = persons_top[i] + persons_bottom[i]                    
                for j, person in enumerate(persons):
                    if len(person[0]) > 0:
                        person_bbox = list(person[0])
                        img_res = cv2.rectangle(img_res, (int(person_bbox[0]), int(person_bbox[1])),
                                                (int(person_bbox[2]), int(person_bbox[3])), [255, 0, 0], 2)

                        # transmit person point to minimap
                        person_point = list(person[1])
                        person_point = np.array(person_point, dtype=np.float32).reshape(1, 1, 2)
                        person_point = cv2.perspectiveTransform(person_point, inv_mat)
                        minimap = cv2.circle(minimap, (int(person_point[0, 0, 0]), int(person_point[0, 0, 1])),
                                                           radius=0, color=(255, 0, 0), thickness=80)

                minimap = cv2.resize(minimap, (width_minimap, height_minimap))
                img_res[30:(30 + height_minimap), (width - 30 - width_minimap):(width - 30), :] = minimap
                imgs_res.append(img_res)

        else:    
            imgs_res = imgs_res + frames[scenes[num_scene][0]:scenes[num_scene][1]] 
    return imgs_res        
 
def write(imgs_res, fps, path_output_video):
    height, width = imgs_res[0].shape[:2]
    out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    for num in range(len(imgs_res)):
        frame = imgs_res[num]
        out.write(frame)
    out.release()    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_ball_track_model', default=r"H:\TemprorayCodes\Dr. Klaus\Ball Tracking\TennisProject\models\model_best.pt", type=str, help='path to pretrained model for ball detection')
    parser.add_argument('--path_court_model', default=r"H:\TemprorayCodes\Dr. Klaus\Ball Tracking\TennisProject\models\model_tennis_court_det.pt", type=str, help='path to pretrained model for court detection')
    parser.add_argument('--path_bounce_model', default=r"H:\TemprorayCodes\Dr. Klaus\Ball Tracking\TennisProject\models\ctb_regr_bounce.cbm", type=str, help='path to pretrained model for bounce detection')
    parser.add_argument('--path_input_video', default=r"H:\TemprorayCodes\Dr. Klaus\Ball Tracking\TennisProject\input\test3_2.mp4", type=str, help='path to input video')
    parser.add_argument('--path_output_video', default=r"H:\TemprorayCodes\Dr. Klaus\Ball Tracking\TennisProject\output\test3_2_28nov.mp4", type=str, help='path to output video')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    frames, fps = read_video(args.path_input_video) 
    scenes = scene_detect(args.path_input_video)    

    print('ball detection')
    ball_detector = BallDetector(args.path_ball_track_model, device)
    ball_track = ball_detector.infer_model(frames)

    print('court detection')
    court_detector = CourtDetectorNet(args.path_court_model, device)
    homography_matrices, kps_court = court_detector.infer_model(frames)

    print('person detection')
    person_detector = PersonDetector(device)
    persons_top, persons_bottom = person_detector.track_players(frames, homography_matrices, filter_players=False)

    # # bounce detection
    bounce_detector = BounceDetector(args.path_bounce_model)
    x_ball = [x[0] for x in ball_track]
    y_ball = [x[1] for x in ball_track]

    import pandas as pd

    # Convert lists to pandas Series
    x_ball_series = pd.Series(x_ball, dtype="float")
    y_ball_series = pd.Series(y_ball, dtype="float")

    # Interpolate missing values
    ball_x_interp = x_ball_series.interpolate(method='pchip').fillna(method='bfill')
    ball_y_interp = y_ball_series.interpolate(method='pchip').fillna(method='bfill')




    ball_track = list(zip(ball_x_interp, ball_y_interp))
    bounces, preds = bounce_detector.predict(ball_x_interp, ball_y_interp)
    mori_bounce = bounces


    # ---------------------------------------------------------------------------------------


    # Derivatives
    dx = ball_x_interp.diff()
    dy = ball_y_interp.diff()

    # Speed
    speed = np.sqrt(dx ** 2 + dy ** 2)

    # Detect bounces as local minima in speed
    bounce_candidates = argrelextrema(speed.to_numpy(), np.less, order=3)[0]  # order=3 = sensitivity

    # Detect hits using angle change in velocity vectors
    vx = dx
    vy = dy
    vx1, vy1 = vx.shift(1), vy.shift(1)

    dot_product = vx * vx1 + vy * vy1
    norm1 = np.sqrt(vx ** 2 + vy ** 2)
    norm2 = np.sqrt(vx1 ** 2 + vy1 ** 2)

    # Avoid divide by zero
    cos_angle = dot_product / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1, 1)

    angle_change = np.arccos(cos_angle)  # radians
    hit_candidates = angle_change[angle_change > 1.0].index  # >1 rad (~57 degrees)

    # ---------------------------------------------------------------------------------------

    # Extract values (first column as Series)
    # y = pd.Series(df.iloc[:, 0].astype(float)).reset_index(drop=True)
    # x = y.index.to_numpy()
    # dy = y.diff()  # derivative for classification

    # --- parameters you can tune ---
    Q_LOW, Q_HIGH = 0.20, 0.80  # use only bottom/top 15% of y as hit zones
    MIN_GAP = 8  # frames between events
    BOUNCE_PROM = 30.0  # prominence threshold for bounces
    HIT_PROM = 60.0  # prominence threshold for hits

    # --- detect bounces as strong local minima in y ---
    bounces, _ = find_peaks(-ball_y_interp.values, prominence=BOUNCE_PROM, distance=MIN_GAP)

    # --- detect hits as strong maxima/minima with prominence ---
    hits_max, _ = find_peaks(ball_y_interp.values, prominence=HIT_PROM, distance=MIN_GAP)
    hits_min, _ = find_peaks((-ball_y_interp).values, prominence=HIT_PROM, distance=MIN_GAP)
    hits = np.union1d(hits_max, hits_min)

    # --- filter hits to only top/bottom court zones ---
    y_low = ball_y_interp.quantile(Q_LOW)
    y_high = ball_y_interp.quantile(Q_HIGH)
    hits = [i for i in hits if (ball_y_interp.iloc[i] <= y_low or ball_y_interp.iloc[i] >= y_high)]

    # --- refine classification for bottom-side events ---
    hits_bottom = []
    real_bounces = []

    for idx in bounces:
        if idx + 1 < len(dy):  # avoid last index
            if dy.iloc[idx + 1] > 0:
                real_bounces.append(idx)  # ball goes up again → bounce
            else:
                hits_bottom.append(idx)  # ball keeps going down → hit at bottom

    # update sets
    hits = np.union1d(hits, hits_bottom)
    bounces = np.array(real_bounces)



    # ----------------------------------------------------------------------------------------------

    BOUNCE_GAP = 5  # minimum allowed gap between valid bounces (adjust per FPS)

    filtered_bounces = []
    for b in sorted(mori_bounce):
        if not filtered_bounces or (b - filtered_bounces[-1]) > BOUNCE_GAP:
            filtered_bounces.append(b)
        else:
            print("The bounce has removed: ", b)
    mori_bounce = set(filtered_bounces)



    HIT_GAP = 5  # minimum allowed gap between valid bounces (adjust per FPS)

    filtered_hits= []
    for b in sorted(hits):
        if not filtered_hits or (b - filtered_hits[-1]) > HIT_GAP:
            filtered_hits.append(b)
        else:
            print("The hit has removed: ", b)
    hits = filtered_hits

    # raw_positions = []
    # for i in range(len(ball_track)):
    #     inv_mat = homography_matrices[i]
    #     ball_point = np.array(ball_track[i], dtype=np.float32).reshape(1, 1, 2)
    #     ball_point = cv2.perspectiveTransform(ball_point, inv_mat)
    #     raw_positions.append(tuple(map(float, ball_point[0, 0])))
    #
    # # Step 2: Compute Y shift: 3500 - max_y
    # max_y = max(pos[1] for pos in raw_positions)
    # y_shift = 3500 - max_y
    #
    # # Step 3: Apply shift
    # new_ball_positions = [(x, y + y_shift) for (x, y) in raw_positions]




    # # old/////////////////////////////////////////
    # imgs_res = main(frames, scenes, mori_bounce, hits, ball_track, homography_matrices, kps_court, persons_top, persons_bottom,
    #                 draw_trace=True)
    #
    # write(imgs_res, fps, args.path_output_video)
    #
    #
    #
    # path_csv = args.path_output_video.replace('.mp4', '.csv')
    # write_csv(path_csv, frames, new_ball_positions, mori_bounce, hits, persons_top, persons_bottom, homography_matrices, kps_court)
    # #old //////////////////////////////////////////////////

    print("Generating Minimap Coordinates...")
    proj_ball, proj_top, proj_bot = get_projected_coords(ball_track, persons_top, persons_bottom, homography_matrices)

    # 2. RUN VIDEO GENERATION
    imgs_res = main(frames, scenes, mori_bounce, hits, ball_track, homography_matrices, kps_court, persons_top,
                    persons_bottom, draw_trace=True)
    write(imgs_res, fps, args.path_output_video)

    # 3. WRITE TO CSV (With new data)
    path_csv = args.path_output_video.replace('.mp4', '.csv')

    write_csv(path_csv, frames, ball_track, mori_bounce, hits,
              persons_top, persons_bottom, homography_matrices, kps_court,
              proj_ball, proj_top, proj_bot)  # <--- Pass the new lists here

    print(f"CSV saved to {path_csv}")

    # thresholds = [0.3, 0.4, 0.5]
    #
    # for t in thresholds:
    #     print(f'Processing threshold {t}')
    #     bounce_detector.threshold = t
    #     bounces, preds = bounce_detector.predict(x_ball, y_ball)
    #
    #     imgs_res = main(frames, scenes, bounces, ball_track, homography_matrices, kps_court,
    #                     persons_top, persons_bottom, draw_trace=True)
    #
    #     output_path = args.path_output_video.replace('.mp4', f'_t{int(t * 100)}.mp4')
    #     write(imgs_res, fps, output_path)
    #
    #
    #
    #
    #
