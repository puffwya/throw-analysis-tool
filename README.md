# Computer Vision System for Throwing Mechanics (WIP)

A computer vision and machine learning pipeline for analyzing throwing mechanics using pose estimation, biomechanical feature extraction, and movement modeling.

---

## Overview

This project explores the use of computer vision and machine learning to evaluate athletic throwing mechanics. It processes video data to extract skeletal movement patterns and converts them into structured biomechanical features for analysis and model development.

The goal is to move from raw pose tracking → feature engineering → predictive performance modeling.

---

## Pipeline

### 1. Pose Estimation
- Uses MediaPipe for real-time skeletal tracking
- Extracts key body joint landmarks from video input

### 2. Feature Extraction
- Computes joint angles from skeletal keypoints
- Structures motion data into time-series biomechanical features
- Prepares dataset for machine learning workflows

### 3. Machine Learning (In Progress)
- Developing models for movement classification and performance analysis
- Experimenting with predictive evaluation of throwing mechanics
- Iterating on feature sets for improved signal quality

### 4. Training Infrastructure
- Runs model training and experiments on AWS EC2
- Supports scalable compute for iterative ML development

---

## Tech Stack

`Python` `MediaPipe` `Computer Vision` `Pose Estimation` `Machine Learning` `AWS EC2`

---

## Current Focus

- Improving feature extraction from pose data
- Transitioning from raw motion tracking to predictive modeling
- Building biomechanical evaluation metrics for athletic performance

---

## Status

Actively in development
