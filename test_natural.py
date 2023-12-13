#!/usr/bin/env python3
import os
import json
import argparse

import pandas as pd
import numpy as np
import nltk

from paddleocr import PaddleOCR

from common import *

natural_image_dir = 'images/natural'
transcription_dir = 'transcriptions/raw'
use_gpu = False
support_upsidedown = True
detection_threshold=0.8

def get_adjusted_line_threshold(result, transcript_data, fudge=0.8):
    bbox_array = np.array([r[0] for r in result])
    line_threshold = transcript_data['meta']['line_threshold']

    zone_width = bbox_array[:, :, 0].max() - bbox_array[:, :, 0].min()
    zone_height = bbox_array[:, :, 1].max() - bbox_array[:, :, 1].min()
    print(zone_width, zone_height, zone_width/transcript_data['meta']['zone_width'], zone_height/transcript_data['meta']['zone_height'])

    height_ratio = zone_height/transcript_data['meta']['zone_height']
    line_threshold = int(line_threshold * height_ratio * fudge)

    return line_threshold

def main():
    ocr = PaddleOCR(use_angle_cls=True, lang='en',
                    use_gpu=use_gpu, show_log=False)

    image_file_paths = os.listdir(natural_image_dir)

    score_df = pd.DataFrame(columns=['img_id','distance','distance_nospaces'])

    for i, p in enumerate(image_file_paths):
        img_id = p.split('_')[0]
        img_path = os.path.join(natural_image_dir, p)
        transcript_path = os.path.join(transcription_dir, img_id+'_result.json')

        with open(transcript_path) as f:
            transcript_data = json.load(f)

        result = ocr.ocr(img_path, cls=support_upsidedown)
        assert len(result) == 1
        result = result[0]

        results_df = results2dataframe(result)
        results_df = results_df[results_df.confidence >= detection_threshold]

        line_threshold = get_adjusted_line_threshold(result, transcript_data)
        lines = get_lines(results_df, line_threshold)
        joined_lines = [' '.join(l) for l in lines]

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
    main()

