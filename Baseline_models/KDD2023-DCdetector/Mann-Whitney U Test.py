#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author : Katuz
@File   : Mann-Whitney U Test.py
@Project: KDD2023-DCdetector
@Time   : 2025/4/6 13:18
"""

import numpy as np
from scipy.stats import mannwhitneyu

model1_f1_value = [87.18, 96.60, 97.02, 96.33, 97.94]
model2_f1_value = [85.27, 94.26, 96.24, 96.40, 96.87]

model1_precision = [83.59, 93.69, 95.63, 93.11, 97.14]
model2_precision = [79.99, 92.22, 93.54, 93.09, 97.07]

model1_recall = [91.10, 99.69, 98.92, 99.77, 98.74]
model2_recall = [91.31, 96.40, 99.10, 99.96, 96.68]

f1_value_stat, f1_value_p = mannwhitneyu(model1_f1_value, model2_f1_value)
precision_stat, precision_p = mannwhitneyu(model1_precision, model2_precision)
recall_stat, recall_p = mannwhitneyu(model1_recall, model2_recall)

print(f"F1-value: Ustats = {f1_value_stat}, p value = {f1_value_p}")
print(f"Precision: Ustats = {precision_stat}, p value = {precision_p}")
print(f"Recall: Ustats = {recall_stat}, p value = {recall_p}")
