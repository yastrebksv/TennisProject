import cv2
import numpy as np
from sympy import Line
from scipy.spatial import distance
def line_intersection(line1, line2):
    """
    Find 2 lines intersection point
    """
    l1 = Line((line1[0], line1[1]), (line1[2], line1[3]))
    l2 = Line((line2[0], line2[1]), (line2[2], line2[3]))

    intersection = l1.intersection(l2)
    point = intersection[0].coordinates if len(intersection) > 0 else None
    return point 

def refine_kps(img, x_ct, y_ct, crop_size=40):
    refined_x_ct, refined_y_ct = x_ct, y_ct
    
    img_height, img_width = img.shape[:2]
    x_min = max(x_ct-crop_size, 0)
    x_max = min(img_height, x_ct+crop_size)
    y_min = max(y_ct-crop_size, 0)
    y_max = min(img_width, y_ct+crop_size)

    img_crop = img[x_min:x_max, y_min:y_max]
    lines = detect_lines(img_crop)
    
    if len(lines) > 1:
        lines = merge_lines(lines)
        if len(lines) == 2:
            inters = line_intersection(lines[0], lines[1])
            if inters:
                new_x_ct = int(inters[1])
                new_y_ct = int(inters[0])
                if new_x_ct > 0 and new_x_ct < img_crop.shape[0] and new_y_ct > 0 and new_y_ct < img_crop.shape[1]:
                    refined_x_ct = x_min + new_x_ct
                    refined_y_ct = y_min + new_y_ct                    
    return refined_y_ct, refined_x_ct

def detect_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)[1]
    lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 30, minLineLength=10, maxLineGap=30)
    lines = np.squeeze(lines) 
    if len(lines.shape) > 0:
        if len(lines) == 4 and not isinstance(lines[0], np.ndarray):
            lines = [lines]
    else:
        lines = []
    return lines

def merge_lines(lines):
    lines = sorted(lines, key=lambda item: item[0])
    mask = [True] * len(lines)
    new_lines = []

    for i, line in enumerate(lines):
        if mask[i]:
            for j, s_line in enumerate(lines[i + 1:]):
                if mask[i + j + 1]:
                    x1, y1, x2, y2 = line
                    x3, y3, x4, y4 = s_line
                    dist1 = distance.euclidean((x1, y1), (x3, y3))
                    dist2 = distance.euclidean((x2, y2), (x4, y4))
                    if dist1 < 20 and dist2 < 20:
                        line = np.array([int((x1+x3)/2), int((y1+y3)/2), int((x2+x4)/2), int((y2+y4)/2)])
                        mask[i + j + 1] = False
            new_lines.append(line)  
    return new_lines       

