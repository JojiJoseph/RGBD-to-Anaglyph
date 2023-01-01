import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from collections import deque

arg_parser = argparse.ArgumentParser("Converts rgbd data to anaglyph")
arg_parser.add_argument("-i","--input-image",type=str,default=None,help="Input rgb image")
arg_parser.add_argument("-d","--depth-image",type=str,default=None,help="Input depth image. Each pixel should be in mm")
arg_parser.add_argument("-D","--distance-between-eyes",type=float,default=0.05,help="Distance between eyes in m")
arg_parser.add_argument("-f","--focal-length",type=float,default=525,help="Focal length in pixels")
arg_parser.add_argument("-cx","--centre-x",type=float,default=319.5,help="cx")
arg_parser.add_argument("-cy","--centre-y",type=str,default=239.5,help="cy")
arg_parser.add_argument("-opt","--optimize",default=False,action="store_true", help="Optimize")

args = arg_parser.parse_args()
if not args.input_image or not args.depth_image:
    print("Please give input rgb image and depth image")
    exit()

img_left = Image.open(args.input_image)
img_left = np.asarray(img_left)

img_depth = Image.open(args.depth_image)
img_depth = np.asarray(img_depth) / 1000. # Normalize to meters

height, width = img_depth.shape

f = args.focal_length
cx = args.centre_x
cy = args.centre_y
distance_between_eyes = args.distance_between_eyes

def deproj(x, y, Z):
    X = (x-cx)*Z/f
    Y = (y-cy)*Z/f
    Z = Z
    return X, Y, Z

print("Generating right eye view...")

def construct_right_image(img_left, depth_image):
    height, width = depth_image.shape
    img_right = np.ones((height, width, 3),np.uint8) *-1
    for y in tqdm(range(height)):
        for x in range(width):
            Z = img_depth[y, x]
            if Z == 0:
                continue
            X, Y, Z = deproj(x, y, Z)
            X_ = X-distance_between_eyes
            x_ = f*X_ + cx*Z
            y_ = f*Y + cy*Z
            x_ /= Z
            y_ /= Z
            if round(x_) < width and round(y_) < height:
                # print(y_,x_,y,x)
                x_ = round(x_)
                y_ = round(y_)
                img_right[y_, x_] = img_left[y,x]
    return img_right

img_right = construct_right_image(img_left, img_depth)
# img_right = np.zeros((height, width, 3),np.uint8)
# for y in tqdm(range(height)):
#     for x in range(width):
#         Z = img_depth[y, x]
#         if Z == 0:
#             continue
#         X, Y, Z = deproj(x, y, Z)
#         X_ = X-distance_between_eyes
#         x_ = f*X_ + cx*Z
#         y_ = f*Y + cy*Z
#         x_ /= Z
#         y_ /= Z
#         if round(x_) < width and round(y_) < height:
#             # print(y_,x_,y,x)
#             x_ = round(x_)
#             y_ = round(y_)
#             img_right[y_, x_] = img_left[y,x]

print("Optimizing right eye view...")
if args.optimize:
    for x in tqdm(range(width)):
        for y in range(height):
            # print(y)
            if (img_right[y, x,:] != [-1, -1, -1]).all():
                continue
            visited = set()
            visited.add((x,y))
            q = deque()
            q.append([x,y])
            while q:
                x_, y_ = q.popleft()
                for i, j in [(0,1),(0,-1),(1,0),(-1,0),(-1,1),(1,-1),(1,1),(-1,-1)]:
                    if 0 <= x_+i < width and 0 <= y_+j < height and (x_+i,y_+j) not in visited:
                        if (img_right[y_+j,x_+i] != [-1,-1,-1]).all():
                            img_right[y, x] = img_right[y_+j,x_+i]
                            q.clear()
                            break
                        elif (x_+i,y_+j) not in visited:
                            visited.add((x_+i,y_+j))
                            q.append([y_+j,x_+i])
img_right = np.clip(img_right, 0, 255).astype(np.uint8)
img_right = cv2.medianBlur(img_right,3)
img_left = img_left.copy()
img_left[:,:,1:] = 0
img_right[:,:,0] = 0
img_3d = img_left + img_right
cv2.namedWindow("3d", cv2.WINDOW_NORMAL)
cv2.imshow("3d",img_3d[:,:,::-1])
cv2.waitKey()

