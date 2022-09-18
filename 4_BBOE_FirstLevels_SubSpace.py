#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from scipy import stats
import nibabel as nib
import seaborn as sns
import matplotlib.pyplot as plt
import nilearn as nl
import glob as glob
from nilearn import image
from nilearn import plotting
from nilearn.glm.first_level import FirstLevelModel
import os
from os import path
from nilearn.image import resample_img
import sys


def CreateBidsStandard(df):
    n_trials = len(df)
    bids_dict = {"onset": [], "duration": [], "trial_type": [], "video": []}
    # video
    bids_dict["onset"].extend(df.vid_start.astype(float))
    bids_dict["duration"].extend(np.abs(df.vid_start - df.vid_end))
    bids_dict["trial_type"].extend(["video"] * n_trials)
    bids_dict["video"].extend(df.video.values)
    bids_dict["onset"].extend(df.fixation2_end.astype(float))
    bids_dict["duration"].extend([8] * n_trials)
    bids_dict["trial_type"].extend(["rating"] * n_trials)
    bids_dict["video"].extend(df.video.values)

    bids_df = pd.DataFrame(bids_dict)
    bids_df = bids_df.sort_values("onset")

    return bids_df


def CheckLengthAndReturn(path_list):
    if len(path_list) == 1:
        return path_list[0]
    return "error"


def GetScanInfoFromName(info_string):
    return int(info_string.split("-")[-1])


def GetIndexesOfColumns(regressor_names, keyword="video"):
    return [
        (idx, regressor_name)
        for idx, regressor_name in enumerate(column_names)
        if keyword in str(regressor_name)
    ]


def CreateConfoundsDf(confounds_df):
    # get columns of interest
    motion_outliers = list(
        filter(lambda x: "motion_outlier" in x, confounds_df.columns)
    )
    xyz = list(
        filter(
            lambda x: ("_y" in x) or ("_x" in x) or ("_z" in x), confounds_df.columns
        )
    )
    csf_wm = list(
        filter(lambda x: ("csf" in x) or ("white_matter" in x), confounds_df.columns)
    )

    ###Do we need to remove non steady state runs?
    non_steady_state = list(
        filter(lambda x: "non_steady_state" in x, confounds_df.columns)
    )

    # filter columns based on columns of interest
    confounds_cols_of_interest = csf_wm + xyz + motion_outliers + non_steady_state
    confounds_df = confounds_df[confounds_cols_of_interest]

    # first_trs
    non_steady_state = np.zeros((600, 5))
    non_steady_state[:5, :5] = np.eye(5)
    confounds_df["1st_tr"] = non_steady_state[:, 0]
    confounds_df["2nd_tr"] = non_steady_state[:, 1]
    confounds_df["3rd_tr"] = non_steady_state[:, 2]
    confounds_df["4th_tr"] = non_steady_state[:, 3]
    confounds_df["5th_tr"] = non_steady_state[:, 4]

    # fill any NANs for modeling
    confounds_df = confounds_df.fillna(0)

    return confounds_df


def SaveVideoRegressors(
    column_names, save_dir, fmri_glm, sub_num, ses_num, run_num, fwhm
):

    idx_vids = GetIndexesOfColumns(column_names)
    for idx, vid in idx_vids:
        contrast = np.zeros((1, design_matrix_df.shape[1]))
        contrast[0, idx] = 1
        z_map = fmri_glm.compute_contrast(contrast, output_type="z_score")
        save_name = "Participant-{}_session-{}_run-{}_fwhm-{}_beta_{}.nii.gz".format(
            sub_num, ses_num, run_num, fwhm, vid
        )
        nib.save(z_map, os.path.join(save_dir, save_name))


def SaveVideoTimeseries(
    column_names, save_dir_ts, denoised, sub_num, ses_num, run_num, fwhm
):
    # idx_vids = GetIndexesOfColumns(column_names)
    # for idx,vid in idx_vids:
    save_name_ts = "Participant-{}_session-{}_run-{}_fwhm-{}_timeseries.nii.gz".format(
        sub_num, ses_num, run_num, fwhm
    )
    nib.save(denoised, os.path.join(save_dir_ts, save_name_ts))


base_dir_path = "/work/abslab/BBOE"
subject_num = sys.argv[1]


brain_space = "ses-1_desc"
space_string = "space-T1w"
fwhm_list = [0, 2, 5, 8]
resampled_brain_mask = -1
mask_string = f"*{brain_space}*brain_mask.nii.gz"
anat_path = os.path.join(
    base_dir_path, f"fmriprep/sub-Participant0{subject_num}/*/anat/"
)
brain_mask_paths = glob.glob(os.path.join(anat_path, mask_string))
brain_mask_path = CheckLengthAndReturn(brain_mask_paths)
brain_mask = nib.load(brain_mask_path)

for fwhm in fwhm_list:

    # func_data_path = os.path.join(base_dir_path, f'fmriprep/sub-Participant0{subject_num}/*/func/')
    func_data_path = os.path.join(
        base_dir_path, f"fmriprep/sub-Participant0{subject_num}/ses-1/func/"
    )
    bold_paths = glob.glob(func_data_path + "*T1w*desc-preproc_bold.nii.gz")

    scan_length = 480  ##8 min scan

    save_dir = os.path.join(
        base_dir_path,
        f"FirstLevels/sub-Participant0{subject_num}/Betas_Smoothed_{fwhm}mm/T1w/",
    )
    save_dir_ts = os.path.join(
        base_dir_path,
        f"FirstLevels/sub-Participant0{subject_num}/Timeseries_Smoothed_{fwhm}mm/T1w/",
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir_ts):
        os.makedirs(save_dir_ts)

    for image_path in bold_paths:
        run_info_from_name = image_path.split("/")[-1].split("_")
        # sub_num = GetScanInfoFromName(run_info_from_name[0])
        ses_num = GetScanInfoFromName(run_info_from_name[1])
        run_num = GetScanInfoFromName(run_info_from_name[3])

        log_file_string = f"*Task_subject-{subject_num}_sess-{ses_num}_run-{run_num}*"
        run_log_file_string = os.path.join(
            base_dir_path,
            "Behavioral",
            f"*0{subject_num}*",
            f"Session_00{ses_num}*",
            log_file_string,
        )
        run_log_files = glob.glob(run_log_file_string)
        run_log_file = CheckLengthAndReturn(run_log_files)

        df = pd.read_csv(run_log_file, delim_whitespace=True)
        bids_df = CreateBidsStandard(df)
        bids_df.video = [
            video_str.split("_")[0].split(".")[0] for video_str in bids_df.video
        ]
        video_names = (
            bids_df[bids_df.trial_type == "video"].trial_type
            + "-"
            + bids_df[bids_df.trial_type == "video"].video
        )
        bids_df.loc[bids_df.trial_type == "video", "trial_type"] = video_names
        bids_df = bids_df.dropna()

        # get confounds
        confound_string = f"sub-Participant0{subject_num}_ses-{ses_num}*run-{run_num}_desc-confounds_timeseries.tsv"
        confound_paths = glob.glob(func_data_path + confound_string)
        confound_path = CheckLengthAndReturn(confound_paths)

        confounds_df = pd.read_csv(confound_path, sep="\t")
        confounds_df = CreateConfoundsDf(confounds_df)

        print(image_path.split("/")[-1])
        tr = 480 / len(confounds_df)

        # load scan
        scan = nl.image.load_img(image_path)

        # resample brain mask
        if resampled_brain_mask == -1:
            affine_func, shape_func = scan.affine, scan.shape
            print("Resampling brain mask")
            resampled_brain_mask = resample_img(
                brain_mask, affine_func, shape_func[:3], interpolation="nearest"
            )

        # smooth, mask, denoise timeseries
        smoothed = nl.image.smooth_img(scan, fwhm)
        denoised = nl.image.clean_img(
            smoothed,
            detrend=False,
            standardize=False,
            confounds=confounds_df,
            high_pass=0.01,
            t_r=tr,
            mask_img=resampled_brain_mask,
        )

        # first level model
        fmri_glm = FirstLevelModel(
            t_r=tr,
            noise_model="ar1",
            standardize=False,
            hrf_model="spm",
            drift_model="cosine",
            verbose=1,
            mask_img=resampled_brain_mask,
        )

        fmri_glm.fit(denoised, events=bids_df)

        design_matrix = fmri_glm.design_matrices_
        design_matrix_df = pd.DataFrame(design_matrix[0])
        # sns.heatmap(design_matrix_df, vmin=-2, vmax=4)
        # plt.show()

        column_names = design_matrix_df.columns.values

        # save timeseries
        SaveVideoTimeseries(
            column_names, save_dir_ts, denoised, subject_num, ses_num, run_num, fwhm
        )

        # save betas
        SaveVideoRegressors(
            column_names, save_dir, fmri_glm, subject_num, ses_num, run_num, fwhm
        )
