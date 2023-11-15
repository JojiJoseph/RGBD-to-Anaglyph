import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from collections import deque, namedtuple
import time

def deproj(x, y, Z, params):
    X = (x-params.cx)*Z/params.f
    Y = (y-params.cy)*Z/params.f
    Z = Z
    return X, Y, Z

def construct_right_image_vectorized(img_left, depth_image, params):
    height, width = depth_image.shape
    mask = 255*np.ones((height,width)).astype(np.uint8)
    img_right = np.ones((img_left.shape),np.uint8) *-1
    x = np.arange(width)
    y = np.arange(height)
    X_left, Y_left = np.meshgrid(x, y)
    Z = depth_image
    X, Y, Z = deproj(X_left, Y_left, Z, params)
    X_ = X-params.distance_between_eyes
    x_ = params.f*X_ + params.cx*Z
    y_ = params.f*Y + params.cy*Z
    Z[Z==0] = -1
    x_ /= Z
    y_ /= Z
    x_[Z==-1] = X_left[Z==-1]
    y_[Z==-1] = Y_left[Z==-1]
    x_ = np.round(x_).astype(np.int32)
    y_ = np.round(y_).astype(np.int32)
    valid = (0 <= x_) & (x_ < width) & (0 <= y_) & (y_ < height)
    X = X.astype(np.int32)
    Y = Y.astype(np.int32)
    img_right[y_[valid], x_[valid]] = img_left[Y_left[valid], X_left[valid]]
    mask[y_[valid], x_[valid]] = 0
    return img_right, mask


def construct_right_image(img_left, depth_image, params):
    height, width = depth_image.shape
    mask = 255*np.ones((height,width)).astype(np.uint8)
    img_right = np.ones((img_left.shape),np.uint8) *-1
    for y in tqdm(range(height)):
        for x in range(width):
            Z = img_depth[y, x]
            if Z == 0:
                img_right[y, x] = img_left[y,x]
                mask[y,x] = 0
                continue
            X, Y, Z = deproj(x, y, Z, params)
            X_ = X-params.distance_between_eyes
            x_ = params.f*X_ + params.cx*Z
            y_ = params.f*Y + params.cy*Z
            x_ /= Z
            y_ /= Z
            if 0 <= round(x_) < width and 0 <= round(y_) < height:
                x_ = round(x_)
                y_ = round(y_)
                img_right[y_, x_] = img_left[y,x]
                mask[y_,x_] = 0
    return img_right, mask

def optimize(img, mask=None, method="avg"):
    height, width, _ = img.shape
    if mask is not None:
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        return img
    for x in tqdm(range(width)):
        for y in range(height):
            if (img[y, x,:] != [-1, -1, -1]).all():
                continue
            visited = set()
            visited.add((x,y))
            q = deque()
            q.append([x,y])
            good_samples = []
            flag = False
            while q:
                for _ in range(len(q)):
                    if not q:
                        break
                    x_, y_ = q.popleft()
                    for i, j in [(0,1),(0,-1),(1,0),(-1,0),(-1,1),(1,-1),(1,1),(-1,-1)]:
                        if 0 <= x_+i < width and 0 <= y_+j < height and (x_+i,y_+j) not in visited:
                            if (img[y_+j,x_+i] != [-1,-1,-1]).all():
                                if method == "nearest":
                                    img[y, x] = img[y_+j,x_+i]
                                    q.clear()
                                    break
                                good_samples.append(img[y_+j,x_+i])
                                flag = True
                            elif (x_+i,y_+j) not in visited:
                                visited.add((x_+i,y_+j))
                                q.append([y_+j,x_+i])
                if flag:
                    img[y, x] = np.mean(good_samples, axis=0)
                    break
    return img

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("Converts rgbd data to anaglyph")
    arg_parser.add_argument("-i","--input-image",type=str,default=None,help="Input rgb image")
    arg_parser.add_argument("-d","--depth-image",type=str,default=None,help="Input depth image. Each pixel should be in mm")
    arg_parser.add_argument("-D","--distance-between-eyes",type=float,default=0.05,help="Distance between eyes in m")
    arg_parser.add_argument("-f","--focal-length",type=float,default=525,help="Focal length in pixels")
    arg_parser.add_argument("-cx","--centre-x",type=float,default=319.5,help="cx")
    arg_parser.add_argument("-cy","--centre-y",type=float,default=239.5,help="cy")
    arg_parser.add_argument("-opt","--optimize",default=False,action="store_true", help="Optimize")
    arg_parser.add_argument("-fl","--flags",default="a", help="Type of 3d visualization. a - anaglyph, s - side by side, c - cross eye, l - left view, r - right - view. Combine to show more visualizations together. For example asc - show anaglyph, side by side and cross eye")
    arg_parser.add_argument("-nf","--normalization-factor",type=float,default=1000,help="Normalization factor for depth. The raw depth value is divided by this number to convert depth to meters.")
    arg_parser.add_argument("-of","--output-file", type=str, help="Output file name")
    arg_parser.add_argument("-ov","--output-view",help="Type of view to be saved. Should be one of 'a', 'c', 's'.")

    args = arg_parser.parse_args()
    if not args.input_image or not args.depth_image:
        print("Please give input rgb image and depth image")
        exit()

    f = args.focal_length
    cx = args.centre_x
    cy = args.centre_y
    distance_between_eyes = args.distance_between_eyes
    flags = args.flags
    normalization_factor = args.normalization_factor
    output_view = args.output_view
    output_file = args.output_file

    img_left = Image.open(args.input_image)
    img_left = np.asarray(img_left)
    if len(img_left.shape) == 2:
        img_left = cv2.merge([img_left, img_left, img_left])

    img_depth = Image.open(args.depth_image)
    img_depth = np.asarray(img_depth) / normalization_factor # Normalize to meters

    height, width = img_depth.shape


    Params = namedtuple("Params", ["f","cx","cy","distance_between_eyes"])
    params = Params(f, cx, cy, distance_between_eyes)

    print("Generating right eye view...")
    start_time = time.time()
    img_right, mask = construct_right_image_vectorized(img_left, img_depth, params)
    end_time = time.time()
    print("Time taken to generate right eye view: {} seconds".format(end_time-start_time))


    if args.optimize:
        print("Optimizing right eye view...")
        # img_right = optimize(img_right, mask) To use inpainting
        img_right = optimize(img_right, None)
    img_right = np.clip(img_right, 0, 255).astype(np.uint8)
    img_right = cv2.medianBlur(img_right,3)
    img_left = img_left.copy()
    img_sbs = np.concatenate([img_left, img_right], axis=1)
    img_cross = np.concatenate([img_right, img_left], axis=1)
    if "l" in flags:
        cv2.imshow("Left image", img_left[:,:,::-1])
    if "r" in flags:
        cv2.imshow("Right image", img_right[:,:,::-1])
    if "s" in flags:
        cv2.imshow("Side By Side", img_sbs[:,:,::-1])
    if "c" in flags:
        cv2.namedWindow("Cross Eye", cv2.WINDOW_NORMAL)
        cv2.imshow("Cross Eye",img_cross[:,:,::-1])
    img_left[:,:,1:] = 0
    img_right[:,:,0] = 0
    img_3d = img_left + img_right
    if "a" in flags:
        cv2.namedWindow("Anaglyph 3d", cv2.WINDOW_NORMAL)
        cv2.imshow("Anaglyph 3d",img_3d[:,:,::-1])
    cv2.waitKey()
    if output_file is not None:
        if output_view not in ["a", "s", "c", "l", "r"]:
            print("Please specify a valid output view!")
            exit()
        if output_view == "a":
            cv2.imwrite(output_file, img_3d[:,:,::-1])
        if output_view == "s":
            cv2.imwrite(output_file, img_sbs[:,:,::-1])
        if output_view == "c":
            cv2.imwrite(output_file, img_cross[:,:,::-1])
