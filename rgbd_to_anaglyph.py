import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from collections import deque, namedtuple

def deproj(x, y, Z, params):
    # f, cx, cy, _ = params
    X = (x-params.cx)*Z/params.f
    Y = (y-params.cy)*Z/params.f
    Z = Z
    return X, Y, Z


def construct_right_image(img_left, depth_image, params):
    height, width = depth_image.shape
    img_right = np.ones((height, width, 3),np.uint8) *-1
    for y in tqdm(range(height)):
        for x in range(width):
            Z = img_depth[y, x]
            if Z == 0:
                continue
            X, Y, Z = deproj(x, y, Z, params)
            X_ = X-params.distance_between_eyes
            x_ = params.f*X_ + params.cx*Z
            y_ = params.f*Y + params.cy*Z
            x_ /= Z
            y_ /= Z
            if round(x_) < width and round(y_) < height:
                x_ = round(x_)
                y_ = round(y_)
                img_right[y_, x_] = img_left[y,x]
    return img_right

def optimize(img):
    for x in tqdm(range(width)):
        for y in range(height):
            if (img[y, x,:] != [-1, -1, -1]).all():
                continue
            visited = set()
            visited.add((x,y))
            q = deque()
            q.append([x,y])
            while q:
                x_, y_ = q.popleft()
                for i, j in [(0,1),(0,-1),(1,0),(-1,0),(-1,1),(1,-1),(1,1),(-1,-1)]:
                    if 0 <= x_+i < width and 0 <= y_+j < height and (x_+i,y_+j) not in visited:
                        if (img[y_+j,x_+i] != [-1,-1,-1]).all():
                            img[y, x] = img[y_+j,x_+i]
                            q.clear()
                            break
                        elif (x_+i,y_+j) not in visited:
                            visited.add((x_+i,y_+j))
                            q.append([y_+j,x_+i])
    return img

if __name__ == "__main__":
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

    Params = namedtuple("Params", ["f","cx","cy","distance_between_eyes"])
    params = Params(f, cx, cy, distance_between_eyes)

    print("Generating right eye view...")
    img_right = construct_right_image(img_left, img_depth, params)


    print("Optimizing right eye view...")
    if args.optimize:
        img_right = optimize(img_right)
    img_right = np.clip(img_right, 0, 255).astype(np.uint8)
    img_right = cv2.medianBlur(img_right,3)
    img_left = img_left.copy()
    cv2.imshow("Left image", img_left)
    cv2.imshow("Right image", img_right)
    img_sbs = np.concatenate([img_left, img_right], axis=1)
    cv2.imshow("Side By Side", img_sbs)
    img_left[:,:,1:] = 0
    img_right[:,:,0] = 0
    img_3d = img_left + img_right

    cv2.namedWindow("3d", cv2.WINDOW_NORMAL)
    cv2.imshow("3d",img_3d[:,:,::-1])
    cv2.waitKey()
