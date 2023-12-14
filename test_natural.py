#!/usr/bin/env python3
import os
import json
import argparse

import pandas as pd
import numpy as np
import nltk

from paddleocr import PaddleOCR

from common import *

def parse_args():
    parser = argparse.ArgumentParser(
        prog='TestNatural',
        description='Run OCR against low quality captures and compare results')

    parser.add_argument('-i', '--images',
        help='Glob pattern for images, default: "images/natural/*"',
        default='images/natural/*')
    parser.add_argument('-t', '--transcripts',
        help='Location of reference transcripts. Default: "transcriptions/raw"',
        default='transcriptions/raw')
    parser.add_argument('-d', '--detection',
        help='Detection threshold for OCR. Default: 0.6',
        default=0.6, type=float)
    parser.add_argument('-g', '--gpu',
        help='Enable GPU processing. Default: False',
        default=False, action='store_true')
    parser.add_argument('-u', '--no-upsidedown',
        help='Do not detect text at extreme angles, will negatively impact quality of detections. Default: False',
        default=False, action='store_true')
    parser.add_argument('-s', '--suffix',
        help='Suffix that was appended after the image id and underscore in transcript filename. Default: "result"',
        default='result')

    return parser.parse_args()


def main(
    natural_image_dir='images/natural/*',
    transcription_dir='transcriptions/raw',
    use_gpu=False,
    support_upsidedown=True,
    detection_threshold=0.6,
    result_suffix='result',
):
    ocr = PaddleOCR(use_angle_cls=True, lang='en',
                    use_gpu=use_gpu, show_log=False)

    image_file_paths = list_dir(natural_image_dir)

    score_df = pd.DataFrame(columns=['img_id','distance','distance_nospaces'])

    for i, img_path in enumerate(image_file_paths):
        img_id = os.path.basename(img_path).split('_')[0]
        transcript_path = os.path.join(transcription_dir,
                                       f'{img_id}_{result_suffix}.json')

        with open(transcript_path) as f:
            transcript_data = json.load(f)

        result = ocr.ocr(img_path, cls=support_upsidedown)
        assert len(result) == 1
        result = result[0]

        results_df = results2dataframe(result)
        results_df = results_df[results_df.confidence >= detection_threshold]

        lines = get_lines(results_df, line_threshold=None)
        joined_lines = [' '.join(l) for l in lines]
        # ~ for l in joined_lines: print(l)

        distance = nltk.edit_distance(' '.join(joined_lines),
                                      ' '.join(transcript_data['lines'])
                                      )

        distance_nospaces = nltk.edit_distance(
                            ''.join(joined_lines).replace(' ', ''),
                            ''.join(transcript_data['lines']).replace(' ', '')
                            )

        score_df.loc[i] = img_id, distance, distance_nospaces

    print('Scores:\n', score_df)
    print('Score means:\n',score_df.drop(['img_id'], axis=1).mean())

if __name__ == '__main__':
    args = parse_args()
    main(
        natural_image_dir=args.images,
        transcription_dir=args.transcripts,
        use_gpu=args.gpu,
        support_upsidedown= not args.no_upsidedown,
        detection_threshold=args.detection,
        result_suffix=args.suffix,
    )

