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
    parser.add_argument('--csv',
        help='Save detailed score output to csv files. Default: False',
        action='store_true')

    return parser.parse_args()


def collect_scores(ids, ref_transcripts, photo_transcripts):
    metric_funcs = [nltk.scores.precision,
                    nltk.scores.recall,
                    nltk.scores.f_measure,
                    ]
    columns = ['ID', 'Precision', 'Recall', 'F1_Score', 'Weight']
    df_line_scores = pd.DataFrame(columns=columns)
    df_word_scores = pd.DataFrame(columns=columns)

    for i, ID, r_lines, t_lines in zip(range(len(ids)), ids, ref_transcripts, photo_transcripts):
        r_words = ' '.join(r_lines).split()
        t_words = ' '.join(t_lines).split()

        lines_score_row = [ID]
        words_score_row = [ID]

        for score_method in metric_funcs:

            lines_score_row.append(score_method(set(r_lines), set(t_lines)))

            words_score_row.append(score_method(set(r_words), set(t_words)))

        lines_score_row.append(len(set(r_lines)))
        words_score_row.append(len(set(r_words)))

        df_line_scores.loc[i] = lines_score_row
        df_word_scores.loc[i] = words_score_row

    return df_line_scores, df_word_scores


def print_scores(df_line_scores, df_word_scores):

    print('SCORES')
    print('Line based macro average:')
    print(df_line_scores.drop(['ID','Weight'], axis=1).mean())
    print('Line based weighted average:')
    print(df_line_scores.drop(['ID','Weight'], axis=1).apply(
            lambda c: np.average(c, weights=df_line_scores.Weight)))
    print('Word based macro average:')
    print(df_word_scores.drop(['ID','Weight'], axis=1).mean())
    print('Word based weighted average:')
    print(df_word_scores.drop(['ID','Weight'], axis=1).apply(
            lambda c: np.average(c, weights=df_word_scores.Weight)))


def main(
    natural_image_dir='images/natural/*',
    transcription_dir='transcriptions/raw',
    use_gpu=False,
    support_upsidedown=True,
    detection_threshold=0.6,
    result_suffix='result',
    csv_debug=False,
):
    ocr = PaddleOCR(use_angle_cls=True, lang='en',
                    use_gpu=use_gpu, show_log=False)

    image_file_paths = list_dir(natural_image_dir)
    print('Found', len(image_file_paths), 'files to process')

    img_ids = []
    ref_transcripts = []
    photo_transcripts = []

    for img_path in image_file_paths:
        img_id = os.path.basename(img_path).split('_')[0]
        img_ids.append(img_id)
        transcript_path = os.path.join(transcription_dir,
                                       f'{img_id}_{result_suffix}.json')

        result = ocr.ocr(img_path, cls=support_upsidedown)
        assert len(result) == 1
        result = result[0]

        results_df = results2dataframe(result)
        results_df = results_df[results_df.confidence >= detection_threshold]

        lines = get_lines(results_df, line_threshold=None)
        joined_lines = [' '.join(l) for l in lines]
        # ~ for l in joined_lines: print(l)

        with open(transcript_path) as f:
            transcript_data = json.load(f)

        ref_transcripts.append(transcript_data['lines'])
        photo_transcripts.append(joined_lines)


    df_line_scores, df_word_scores = collect_scores(
        img_ids, ref_transcripts, photo_transcripts)

    if csv_debug:
        df_line_scores.to_csv('line_scores.csv')
        df_word_scores.to_csv('word_scores.csv')

    print_scores(df_line_scores, df_word_scores)

if __name__ == '__main__':
    args = parse_args()
    main(
        natural_image_dir=args.images,
        transcription_dir=args.transcripts,
        use_gpu=args.gpu,
        support_upsidedown= not args.no_upsidedown,
        detection_threshold=args.detection,
        result_suffix=args.suffix,
        csv_debug=args.csv,
    )

