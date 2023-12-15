#!/usr/bin/env python3
import glob
import os

import numpy as np
import pandas as pd

def list_dir(glob_pattern):
    r = glob.glob(glob_pattern)
    if len(r) == 1 and os.path.isdir(r[0]):
        # If user passes a non-glob path, convert it to a glob pattern
        r = glob.glob(os.path.join(glob_pattern, '*'))
    return r

def get_lines(result_df, line_threshold=None):
    lines = []
    group_start = 0
    box_y_distances = np.diff(result_df.y)
    if line_threshold is None:
        line_threshold = box_y_distances.mean()

    for i in np.where(box_y_distances > line_threshold)[0]:
        group = result_df.iloc[group_start:i+1]
        lines.append(group.sort_values('x').text.values.tolist())
        group_start = i + 1

    group = result_df.iloc[group_start:]
    lines.append(group.sort_values('x').text.values.tolist())

    return lines

def get_strings(result):
    return [r[1][0] for r in result]

def get_confidences(result):
    return [r[1][1] for r in result]

def results2dataframe(result):
    bbox_array = np.array([r[0] for r in result])
    upper_left_corners = bbox_array[:, 0, :]

    results_df = pd.DataFrame([upper_left_corners[:, 0],
                               upper_left_corners[:, 1],
                               get_strings(result),
                               get_confidences(result)]).T
    results_df.columns = ['x', 'y', 'text', 'confidence']
    results_df.sort_values(by='y', kind='stable', inplace=True)

    return results_df
