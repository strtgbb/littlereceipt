#!/usr/bin/env python3


import os
import json
import argparse

import pandas as pd
import numpy as np

from paddleocr import PaddleOCR

from common import *


def parse_args():
    parser = argparse.ArgumentParser(
        prog='ParseIdeal',
        description='Run OCR against high quality scans to create reference data.')

    parser.add_argument('-i', '--images',
        help='Glob pattern for images, default: "images/ideal/*"',
        default='images/ideal/*')
    parser.add_argument('-t', '--transcripts',
        help='Path to save transcript data to. Default: "transcriptions/raw"',
        default='transcriptions/raw')
    parser.add_argument('-d', '--detection',
        help='Detection threshold for OCR. Default: 0.8',
        default=0.8, type=float)
    parser.add_argument('-g', '--gpu',
        help='Enable GPU processing. Default: False',
        default=False, action='store_true')
    parser.add_argument('-u', '--upsidedown',
        help='Detect text at extreme angles, will have negative impacts on performance and post-processing. Default: False',
        default=False, action='store_true')
    parser.add_argument('-s', '--suffix',
        help='Suffix to append after the image id and underscore in filename. Default: "result"',
        default='result')
    parser.add_argument('--save-raw',
        help='Save the raw output from OCR to output file. Default: False',
        default=False, action='store_true')

    return parser.parse_args()


def save_results(file_path, lines, meta, raw_results):
    with open(file_path, 'w') as f:
        json.dump({'lines': lines,
                   'meta': meta,
                   'raw_results': raw_results,
                   }, f, indent=2)

def main(
    ideal_image_dir = 'images/ideal/*',
    transcription_dir = 'transcriptions/raw',
    use_gpu = False,
    support_upsidedown = False,
    detection_threshold=0.8,
    result_suffix='result',
    save_raw=False,
):
    ocr = PaddleOCR(use_angle_cls=True, lang='en',
                    use_gpu=use_gpu, show_log=False)

    image_file_paths = list_dir(ideal_image_dir)

    for img_path in image_file_paths:
        img_id = os.path.basename(img_path).split('_')[0]
        result = ocr.ocr(img_path, cls=support_upsidedown)
        assert len(result) == 1
        result = result[0]

        bbox_array = np.array([r[0] for r in result])
        zone_width = bbox_array[:, :, 0].max() - bbox_array[:, :, 0].min()
        zone_height = bbox_array[:, :, 1].max() - bbox_array[:, :, 1].min()

        results_df = results2dataframe(result)
        results_df = results_df[results_df.confidence >= detection_threshold]

        lines = get_lines(results_df, line_threshold=None)

        joined_lines = [' '.join(l) for l in lines]

        # ~ for l in joined_lines: print(l)

        meta = {'id': img_id,
                'zone_width':zone_width,
                'zone_height':zone_height,
                'detection_threshold':detection_threshold,
                }

        save_results(os.path.join(transcription_dir,
                                  f'{img_id}_{result_suffix}.json'),
                     joined_lines,
                     meta,
                     result if save_raw else None,
                     )

if __name__ == '__main__':
    args = parse_args()
    main(
        ideal_image_dir=args.images,
        transcription_dir=args.transcripts,
        use_gpu=args.gpu,
        support_upsidedown=args.upsidedown,
        detection_threshold=args.detection,
        result_suffix=args.suffix,
        save_raw=args.save_raw,
    )
