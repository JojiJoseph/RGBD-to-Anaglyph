# RGBD to Anaglyph

A program to convert rgbd image (a pair of rgb image and corresponding depth image) to anaglyph 3d.

## How to Run

User argument `-i` to specify rgb image and `-d` to specify depth image. Depth should be in mm.

```bash
python3 rgbd_to_anaglyph.py -i car.jpg -d car.png -opt

```

Save a file but not showing on the screen. Will be useful for batch processing with a bash script.

```bash
python rgbd_to_anaglyph.py -i car.jpg -d car.png --opt -fl none -of out.png -ov a
```

## Visualizations supported

Use `--flags` or `-fl` to show different types of 3d representations.

    a - anaglyph (default)
    s - side by side
    c - cross eye 3d

Combine to show mutliple 3d representations

    asc - Show anaglyph, side by side and cross eye

## Usage
```
usage: Converts rgbd data to anaglyph [-h] [-i INPUT_IMAGE] [-d DEPTH_IMAGE]
                                      [-D DISTANCE_BETWEEN_EYES]
                                      [-f FOCAL_LENGTH] [-cx CENTRE_X]
                                      [-cy CENTRE_Y] [-opt] [-fl FLAGS]
                                      [-nf NORMALIZATION_FACTOR]
                                      [-of OUTPUT_FILE] [-ov OUTPUT_VIEW]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_IMAGE, --input-image INPUT_IMAGE
                        Input rgb image
  -d DEPTH_IMAGE, --depth-image DEPTH_IMAGE
                        Input depth image. Each pixel should be in mm
  -D DISTANCE_BETWEEN_EYES, --distance-between-eyes DISTANCE_BETWEEN_EYES
                        Distance between eyes in m
  -f FOCAL_LENGTH, --focal-length FOCAL_LENGTH
                        Focal length in pixels
  -cx CENTRE_X, --centre-x CENTRE_X
                        cx
  -cy CENTRE_Y, --centre-y CENTRE_Y
                        cy
  -opt, --optimize      Optimize
  -fl FLAGS, --flags FLAGS
                        Type of 3d visualization. a - anaglyph, s - side by
                        side, c - cross eye, l - left view, r - right - view.
                        Combine to show more visualizations together. For
                        example asc - show anaglyph, side by side and cross
                        eye
  -nf NORMALIZATION_FACTOR, --normalization-factor NORMALIZATION_FACTOR
                        Normalization factor for depth. The raw depth value is
                        divided by this number to convert depth to meters.
  -of OUTPUT_FILE, --output-file OUTPUT_FILE
                        Output file name
  -ov OUTPUT_VIEW, --output-view OUTPUT_VIEW
                        Type of view to be saved. Should be one of 'a', 'c',
                        's'.
```
## TODO

 - [x] Speed up computation (60 times improvement is noticed)
 - [ ] Document functions

## Acknowledgement
Sample images are taken from http://redwood-data.org/3dscan/dataset.html
