# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 15:08:30 2025

@author: ktrpt
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import zipfile
from datetime import datetime


st.title("Upper Limb Stick Picture Viewer")
st.markdown("Upload a CSV file (pose_output.csv) exported from MediaPipe.")

uploaded_file = st.file_uploader("Upload pose_output.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    n_frames = len(df)
    n_landmarks = 33

    pos_data_x = np.zeros((n_frames, n_landmarks))
    pos_data_y = np.zeros((n_frames, n_landmarks))
    for i in range(n_landmarks):
        pos_data_x[:, i] = df[f"x_{i}_px"]
        pos_data_y[:, i] = -df[f"y_{i}_px"]

    mid_head = np.array([(pos_data_x[:, 2] + pos_data_x[:, 5]) / 2, (pos_data_y[:, 2] + pos_data_y[:, 5]) / 2]).T
    l_shoul = np.array([pos_data_x[:, 11], pos_data_y[:, 11]]).T
    l_elb = np.array([pos_data_x[:, 13], pos_data_y[:, 13]]).T
    l_wri = np.array([pos_data_x[:, 15], pos_data_y[:, 15]]).T
    r_shoul = np.array([pos_data_x[:, 12], pos_data_y[:, 12]]).T
    r_elb = np.array([pos_data_x[:, 14], pos_data_y[:, 14]]).T
    r_wri = np.array([pos_data_x[:, 16], pos_data_y[:, 16]]).T

    def calangle(v1, v2):
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.degrees(np.arccos(dot / (norm1 * norm2)))

    vertical = np.array([0, 1])
    horizontal = np.array([1, 0])

    vec_l_abd = l_elb - l_shoul
    vec_l_elev = r_shoul - l_shoul
    l_abd_ang = 180 - np.array([calangle(v, vertical) for v in vec_l_abd])
    l_elev_ang = 180 - np.array([calangle(v, horizontal) for v in vec_l_elev])
    l_max_abd = np.max(np.nan_to_num(l_abd_ang))
    l_max_elev = np.max(np.nan_to_num(l_elev_ang))

    vec_r_abd = r_elb - r_shoul
    vec_r_elev = l_shoul - r_shoul
    r_abd_ang = 180 - np.array([calangle(v, vertical) for v in vec_r_abd])
    r_elev_ang = np.array([calangle(v, horizontal) for v in vec_r_elev])
    r_max_abd = np.max(np.nan_to_num(r_abd_ang))
    r_max_elev = np.max(np.nan_to_num(r_elev_ang))

    origin = mid_head[0]

    def plot_side(fig_title, head, shoul1, elb, wri, shoul2, every=10, side="left", abd=0, elev=0):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-300, 300)
        ax.set_ylim(-300, 300)
        ax.set_aspect('equal')
        ax.set_title(fig_title)
        ax.grid(True, alpha=0.2)

        for frame in range(0, len(head), every):
            h = head[frame] - origin
            s1 = shoul1[frame] - origin
            e = elb[frame] - origin
            w = wri[frame] - origin
            s2 = shoul2[frame] - origin

            ax.scatter(h[0], h[1], color='magenta', s=100, alpha=0.2)
            ax.scatter(s1[0], s1[1], color='gray', s=50, alpha=0.2)
            ax.scatter(e[0], e[1], color='gray', s=50, alpha=0.2)
            ax.scatter(w[0], w[1], color='crimson', s=50, alpha=0.2)
            ax.scatter(s2[0], s2[1], color='gray', s=50, alpha=0.2)

            ax.plot([s1[0], s2[0]], [s1[1], s2[1]], color='gray', alpha=0.2)
            ax.plot([s1[0], e[0]], [s1[1], e[1]], color='gray', alpha=0.2)
            ax.plot([e[0], w[0]], [e[1], w[1]], color='gray', alpha=0.2)

        ax.text(-280, 230, f"Max ABD: {abd:.1f}°", fontsize=12)
        ax.text(-280, 190, f"Max Elevation: {elev:.1f}°", fontsize=12)
        return fig

    fig_l = plot_side("Left Arm Stick Picture", mid_head, l_shoul, l_elb, l_wri, r_shoul, side="left", abd=l_max_abd, elev=l_max_elev)
    fig_r = plot_side("Right Arm Stick Picture", mid_head, r_shoul, r_elb, r_wri, l_shoul, side="right", abd=r_max_abd, elev=r_max_elev)

    st.pyplot(fig_l)
    st.pyplot(fig_r)

    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zip_file:
        buf_l = BytesIO()
        fig_l.savefig(buf_l, format="png")
        zip_file.writestr("left_shoulder_flow.png", buf_l.getvalue())

        buf_r = BytesIO()
        fig_r.savefig(buf_r, format="png")
        zip_file.writestr("right_shoulder_flow.png", buf_r.getvalue())

    st.download_button(
        label="Download ZIP of Both Figures",
        data=zip_buf.getvalue(),
        file_name=f"shoulder_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        mime="application/zip"
    )
