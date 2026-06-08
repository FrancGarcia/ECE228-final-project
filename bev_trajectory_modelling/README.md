# BEV Trajectory Modelling — Tracking & Prediction

Stage 4 of the pipeline: tracking agents across time and predicting their trajectories.

Extends the BEV fusion with **multi-object Kalman tracking**, **ego-motion compensation**, and **crossing-intent prediction**, producing world-frame trajectories and velocity estimates from a moving camera.

## Contents

| File | Description |
|---|---|
| `bev_trajectory_prediction.ipynb` | Initial full pipeline: detection → depth → BEV → ego-compensated Kalman tracking → intent |
| `bev_trajectory_prediction_updated.ipynb` | Updated/refined version of the pipeline |
| `bev_trajectory_prediction_with_validation.ipynb` | Final version with full validation metrics (recommended) |

## Pipeline highlights

- **Ego-motion compensation** — parses KITTI `oxts` GPS/IMU data into world-frame transforms, so a parked car stays still in BEV even as the ego vehicle turns or accelerates.
- **Kalman tracking** — per-object kinematic state estimation (`dt = 0.1 s`), persistent instance IDs, and class-mismatch penalties for robust data association.
- **Crossing intent** — PIE-inspired, colour-coded BEV overlay estimating whether pedestrians/cyclists are likely to cross.
- **Validation** — ground-truth matching plus self-consistency metrics (velocity plausibility, track longevity, position jumps, forecasting accuracy).

## Data requirement

Unlike the single-frame fusion, this stage needs a **KITTI raw drive sequence** including the `oxts/` odometry folder and camera calibration, because tracking and ego-motion compensation operate across consecutive frames.

## Requirements

Google Colab with a GPU (T4 or better). Each notebook lists its execution order in the first cell.
