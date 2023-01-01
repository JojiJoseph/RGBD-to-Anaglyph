# RGBD to Anaglyph

A program to convert rgbd image (a pair of rgb image and corresponding depth image) to anaglyph 3d.

## How to Run

User argument `-i` to specify rgb image and `-d` to specify depth image. Depth should be in mm.

```bash
python3 rgbd_to_anaglyph.py -i car.jpg -d car.png -opt

```

## Visualizations supported

Use `--flags` or `-fl` to show different types of 3d representations.

    a - anaglyph (default)
    s - side by side
    c - cross eye 3d

Combine to show mutliple 3d representations

    asc - Show anaglyph, side by side and cross eye

## TODO

 - [ ] Speed up computation
 - [ ] Document functions

## Acknowledgement
Sample images are taken from http://redwood-data.org/3dscan/dataset.html
