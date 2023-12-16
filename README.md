# LittleReceipt

This project contains scripts and demo data for comparing OCR scans under ideal
conditions to photos snapped in casual conditions.

Currently it supports only
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/README_en.md)

The example data is focused around receipts.


## Quickstart

To run against the included data `images/ideal` and `images/natural`

To install dependencies (venv recommended):

`pip install -r requirements.txt`

To create the transcript files from the ideal images run

`./parse ideal.py`

To run a comparison of the natural images to the transcripts run

`./test_natural.py`


## A note on file names
The portion of the filename before an underscore `_` is taken as the image's ID.

Ex. The file `12_scan.png` would be processed as having id = 12.

The scan and photo of each receipt must have the same id.


## Processing new data
To create the transcript files from new images, pass their directory as a glob pattern

`./parse ideal.py --images path/to/images/* --transcripts save/transcripts/here`

At this stage you can optionally correct the transcripts manually.

To evaluate the photos against the transcripts run

`./test_natural.py --images path/to/images/*  --transcripts path/to/transcripts`

To see more options for controlling the scripts use

`./parse ideal.py -h` and `./test_natural.py -h`

