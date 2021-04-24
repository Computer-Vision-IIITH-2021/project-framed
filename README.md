# WCT2
This repository contains PyTorch implementation of the paper [Photorealistic Style Transfer via Wavelet Transforms](https://arxiv.org/abs/1903.09760).
- Avinash Prabhu - 2018102027
- Fiza Husain - 2018101035
- Mallika Subramanian - 2018101041
- Tanvi Karandikar - 2018101059

## How to run the code
```python
cd src/
python main.py
```
### Arguments
- `--content`: FOLDER-PATH-TO-CONTENT-IMAGES
- `--content_segment`: FOLDER-PATH-TO-CONTENT-SEGMENT-LABEL-IMAGES
- `--style`: FOLDER-PATH-TO-STYLE-IMAGES
- `--style_segment`: FOLDER-PATH-TO-STYLE-SEGMENT-LABEL-IMAGES
- `--output`: FOLDER-PATH-TO-OUTPUT-IMAGES
- `--image_size`: output image size
- `--alpha`: alpha determines the blending ratio between content and stylized features

