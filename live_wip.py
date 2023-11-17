import cv2
import torch
import urllib.request
import matplotlib.pyplot as plt
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
urllib.request.urlretrieve(url, filename)
# model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("img", img)
cv2.waitKey(0)

cap = cv2.VideoCapture("video.mp4")

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from collections import deque, namedtuple
import time
import jax

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

# img = None
frame = img.copy()
paused = False
def optimize(img, mask=None, method="avg"):
    height, width, _ = img.shape
    # if mask is not None:
    img = np.clip(img, 0, 255).astype(np.uint8)
    # img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    # img2 = cv2.filter2D(img, -1, np.ones((9, 9),np.float32)/81.)
    img2 = cv2.medianBlur(img, 9)
    img[mask] = img2[mask]
    return img
    # for x in tqdm(range(width)):
    #     for y in range(height):
    #         if (img[y, x,:] != [-1, -1, -1]).all():
    #             continue
    #         visited = set()
    #         visited.add((x,y))
    #         q = deque()
    #         q.append([x,y])
    #         good_samples = []
    #         flag = False
    #         while q:
    #             for _ in range(len(q)):
    #                 if not q:
    #                     break
    #                 x_, y_ = q.popleft()
    #                 for i, j in [(0,1),(0,-1),(1,0),(-1,0),(-1,1),(1,-1),(1,1),(-1,-1)]:
    #                     if 0 <= x_+i < width and 0 <= y_+j < height and (x_+i,y_+j) not in visited:
    #                         if (img[y_+j,x_+i] != [-1,-1,-1]).all():
    #                             if method == "nearest":
    #                                 img[y, x] = img[y_+j,x_+i]
    #                                 q.clear()
    #                                 break
    #                             good_samples.append(img[y_+j,x_+i])
    #                             flag = True
    #                         elif (x_+i,y_+j) not in visited:
    #                             visited.add((x_+i,y_+j))
    #                             q.append([y_+j,x_+i])
    #             if flag:
    #                 img[y, x] = np.mean(good_samples, axis=0)
    #                 break
    # return img
while True:
    if not paused:
        ret, frame = cap.read()
    if not ret:
        break
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()

    # plt.imshow(output)
    # plt.show()
    output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    Params = namedtuple("Params", ["f","cx","cy","distance_between_eyes"])
    params = Params(525, frame.shape[1]//2, frame.shape[0]//2, 0.05)

    print("Generating right eye view...")
    start_time = time.time()
    depth = (256-output)/50.
    # depth[:,:] = 0
    print(depth.min(), depth.max())
    img_right, mask = (construct_right_image_vectorized)(frame, depth, params)
    img_right = img_right.astype(np.uint8)
    img_right[mask==255] = [0,0,0]
    img_right = optimize(img_right, mask)
    print("img_right", img_right.min(), img_right.max(), img_right.dtype, img_right.shape, img_right.dtype)
    end_time = time.time()
    print("Time taken to generate right eye view: {} seconds".format(end_time-start_time))
    output = cv2.cvtColor(255-output, cv2.COLOR_GRAY2BGR)
    cv2.imshow('output', output)
    cv2.imshow('input', img[...,::-1])
    cv2.imshow('right', img_right)
    img_left = frame.copy()
    img_left[:,:,:-1] = 0
    img_right[:,:,-1] = 0
    img_3d = img_left + img_right
    # cv2.imshow()
    cv2.namedWindow("img_3d", cv2.WINDOW_NORMAL)
    cv2.imshow("img_3d", img_3d)
    key = cv2.waitKey(1) & 0xFF
    if key in [ord('q'), 27]:
        break
    elif key == ord('p'):
        paused = not paused

cv2.destroyAllWindows()