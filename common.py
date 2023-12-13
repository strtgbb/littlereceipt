#!/usr/bin/env python3
import numpy as np
import pandas as pd

def get_lines(result_df, line_threshold=30):
    lines = []
    group_start = 0
    box_y_distances = np.diff(result_df.y)
    for i in np.where(box_y_distances > line_threshold)[0]:
        group = result_df.iloc[group_start:i+1]
        lines.append(group.sort_values('x').text.values)
        group_start = i + 1

    group = result_df.iloc[group_start:]
    lines.append(group.sort_values('x').text.values)

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
