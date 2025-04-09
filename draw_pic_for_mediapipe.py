# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 15:08:30 2025

@author: ktrpt
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tkinter import Tk, filedialog

# 日本語フォント対応（Windows想定）
plt.rcParams['font.family'] = 'MS Gothic'

# CSVファイル選択ダイアログ
Tk().withdraw()
csv_path = filedialog.askopenfilename(title="CSVファイルを選択", filetypes=[("CSV files", "*.csv")])
df = pd.read_csv(csv_path)

# 角度計算用関数
def calangle(v1, v2):
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.degrees(np.arccos(dot / (norm1 * norm2)))

# 座標データ整形
n_frames = len(df)
n_landmarks = 33
pos_data_x = np.zeros((n_frames, n_landmarks))
pos_data_y = np.zeros((n_frames, n_landmarks))
for i in range(n_landmarks):
    pos_data_x[:, i] = df[f"x_{i}_px"]
    pos_data_y[:, i] = -df[f"y_{i}_px"]  # Y軸反転（上が正）

# ランドマーク設定
mid_head = np.array([(pos_data_x[:, 2] + pos_data_x[:, 5]) / 2, (pos_data_y[:, 2] + pos_data_y[:, 5]) / 2]).T
l_shoul = np.array([pos_data_x[:, 11], pos_data_y[:, 11]]).T
l_elb = np.array([pos_data_x[:, 13], pos_data_y[:, 13]]).T
l_wri = np.array([pos_data_x[:, 15], pos_data_y[:, 15]]).T
r_shoul = np.array([pos_data_x[:, 12], pos_data_y[:, 12]]).T
r_elb = np.array([pos_data_x[:, 14], pos_data_y[:, 14]]).T
r_wri = np.array([pos_data_x[:, 16], pos_data_y[:, 16]]).T

# 角度計算
vertical = np.array([0, 1])
horizontal = np.array([1, 0])
vec_l_abd = l_elb - l_shoul
vec_l_elev = r_shoul - l_shoul
l_abd_ang = 180 - np.array([calangle(v, vertical) for v in vec_l_abd])
l_elev_ang = 180 - np.array([calangle(v, horizontal) for v in vec_l_elev])
l_abd_ang = np.nan_to_num(l_abd_ang)
l_elev_ang = np.nan_to_num(l_elev_ang)
l_max_abd = np.max(l_abd_ang)
l_max_elev = np.max(l_elev_ang)

vec_r_abd = r_elb - r_shoul
vec_r_elev = l_shoul - r_shoul
r_abd_ang = 180 - np.array([calangle(v, vertical) for v in vec_r_abd])
r_elev_ang = np.array([calangle(v, horizontal) for v in vec_r_elev])
r_abd_ang = np.nan_to_num(r_abd_ang)
r_elev_ang = np.nan_to_num(r_elev_ang)
r_max_abd = np.max(r_abd_ang)
r_max_elev = np.max(r_elev_ang)

# 描画用フォルダ作成
os.makedirs("Fig", exist_ok=True)

# --- 左上肢の動作フロー図を描画 ---
fig_l, ax_l = plt.subplots(figsize=(6, 6))
ax_l.set_xlim(-300, 300)
ax_l.set_ylim(-300, 300)
ax_l.set_aspect('equal')
ax_l.set_title("左上肢スティックピクチャ（連続描画）")
ax_l.grid(True, alpha=0.2)

origin = mid_head[0]
for frame in range(0, n_frames, 10):
    head = mid_head[frame] - origin
    ls = l_shoul[frame] - origin
    le = l_elb[frame] - origin
    lw = l_wri[frame] - origin
    rs = r_shoul[frame] - origin

    ax_l.scatter(head[0], head[1], color='magenta', s=100, alpha=0.2)
    ax_l.scatter(ls[0], ls[1], color='gray', s=50, alpha=0.2)
    ax_l.scatter(le[0], le[1], color='gray', s=50, alpha=0.2)
    ax_l.scatter(lw[0], lw[1], color='crimson', s=50, alpha=0.2)
    ax_l.scatter(rs[0], rs[1], color='gray', s=50, alpha=0.2)

    ax_l.plot([ls[0], rs[0]], [ls[1], rs[1]], color='gray', alpha=0.2)
    ax_l.plot([ls[0], le[0]], [ls[1], le[1]], color='gray', alpha=0.2)
    ax_l.plot([le[0], lw[0]], [le[1], lw[1]], color='gray', alpha=0.2)

plt.figtext(0.15, 0.03, f"最大肩外転 = {l_max_abd:.1f}°", fontsize=12)
plt.figtext(0.15, 0.00, f"最大肩挙上 = {l_max_elev:.1f}°", fontsize=12)
plt.tight_layout()
fig_l.savefig("Fig/left_shoulder_flow.png")
plt.close(fig_l)

# --- 右上肢の動作フロー図を描画 ---
fig_r, ax_r = plt.subplots(figsize=(6, 6))
ax_r.set_xlim(-300, 300)
ax_r.set_ylim(-300, 300)
ax_r.set_aspect('equal')
ax_r.set_title("右上肢スティックピクチャ（連続描画）")
ax_r.grid(True, alpha=0.2)

for frame in range(0, n_frames, 10):
    head = mid_head[frame] - origin
    rs = r_shoul[frame] - origin
    re = r_elb[frame] - origin
    rw = r_wri[frame] - origin
    ls = l_shoul[frame] - origin

    ax_r.scatter(head[0], head[1], color='magenta', s=100, alpha=0.2)
    ax_r.scatter(rs[0], rs[1], color='gray', s=50, alpha=0.2)
    ax_r.scatter(re[0], re[1], color='gray', s=50, alpha=0.2)
    ax_r.scatter(rw[0], rw[1], color='crimson', s=50, alpha=0.2)
    ax_r.scatter(ls[0], ls[1], color='gray', s=50, alpha=0.2)

    ax_r.plot([ls[0], rs[0]], [ls[1], rs[1]], color='gray', alpha=0.2)
    ax_r.plot([rs[0], re[0]], [rs[1], re[1]], color='gray', alpha=0.2)
    ax_r.plot([re[0], rw[0]], [re[1], rw[1]], color='gray', alpha=0.2)

plt.figtext(0.15, 0.03, f"最大肩外転 = {r_max_abd:.1f}°", fontsize=12)
plt.figtext(0.15, 0.00, f"最大肩挙上 = {r_max_elev:.1f}°", fontsize=12)
plt.tight_layout()
fig_r.savefig("Fig/right_shoulder_flow.png")
plt.close(fig_r)

print("左右上肢の連続スティックピクチャを保存しました → Fig/left_shoulder_flow.png / Fig/right_shoulder_flow.png")
