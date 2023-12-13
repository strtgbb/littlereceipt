#!/usr/bin/env python3


import os
import json
import argparse

import pandas as pd
import numpy as np

from paddleocr import PaddleOCR

from common import *


ideal_image_dir = 'images/ideal'
transcription_dir = 'transcriptions/raw'
use_gpu = False
support_upsidedown = False
line_threshold=30
detection_threshold=0.8


def save_results(file_path, lines, meta, raw_results):
    with open(file_path, 'w') as f:
        json.dump({'lines': lines,
                   'meta': meta,
                   'raw_results': raw_results,
                   }, f, indent=2)

def main():
    ocr = PaddleOCR(use_angle_cls=True, lang='en',
                    use_gpu=use_gpu, show_log=False)

    image_file_paths = os.listdir(ideal_image_dir)

    for p in image_file_paths:
        img_id = p.split('_')[0]
        img_path = os.path.join(ideal_image_dir, p)
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

        meta = {'zone_width':zone_width,
                'zone_height':zone_height,
                'line_threshold':line_threshold,
                'detection_threshold':detection_threshold,
                }

        save_results(os.path.join(transcription_dir, img_id+'_result.json'),
                     joined_lines, meta, result)

if __name__ == '__main__':
    main()
