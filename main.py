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

def main(frames, scenes, bounces, ball_track, homography_matrices, kps_court, persons_top, persons_bottom,
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
    is_track = [x is not None for x in homography_matrices] 
    for num_scene in range(len(scenes)):
        sum_track = sum(is_track[scenes[num_scene][0]:scenes[num_scene][1]])
        len_track = scenes[num_scene][1] - scenes[num_scene][0]

        eps = 1e-15
        scene_rate = sum_track/(len_track+eps)
        if (scene_rate > 0.5):
            court_img = get_court_img()

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

                # draw bounce in minimap
                if i in bounces and inv_mat is not None:
                    ball_point = ball_track[i]
                    ball_point = np.array(ball_point, dtype=np.float32).reshape(1, 1, 2)
                    ball_point = cv2.perspectiveTransform(ball_point, inv_mat)
                    court_img = cv2.circle(court_img, (int(ball_point[0, 0, 0]), int(ball_point[0, 0, 1])),
                                                       radius=0, color=(0, 255, 255), thickness=50)

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
    parser.add_argument('--path_ball_track_model', type=str, help='path to pretrained model for ball detection')
    parser.add_argument('--path_court_model', type=str, help='path to pretrained model for court detection')
    parser.add_argument('--path_bounce_model', type=str, help='path to pretrained model for bounce detection')
    parser.add_argument('--path_input_video', type=str, help='path to input video')
    parser.add_argument('--path_output_video', type=str, help='path to output video')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    # bounce detection
    bounce_detector = BounceDetector(args.path_bounce_model)
    x_ball = [x[0] for x in ball_track]
    y_ball = [x[1] for x in ball_track]
    bounces = bounce_detector.predict(x_ball, y_ball)

    imgs_res = main(frames, scenes, bounces, ball_track, homography_matrices, kps_court, persons_top, persons_bottom,
                    draw_trace=True)

    write(imgs_res, fps, args.path_output_video)





