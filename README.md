# Real-Time Sign Detection

**Author:** Sanjana Madhu

## Overview

This project uses a Convolutional Neural Network (CNN) to recognize American Sign Language (ASL) gestures from a webcam feed and convert them into text in real time. It supports the full English alphabet (`A–Z`) along with `space`, `del`, and `nothing` tokens for building words interactively.

## Features

- Real-time hand gesture detection using MediaPipe (`cvzone`)
- Image preprocessing: grayscale, resized to 64x64
- Letter prediction with stability and confidence checks
- Word formation logic using live gesture input
- Live feedback via OpenCV window

## Dataset & Model

- **Dataset**: [ASL Alphabet – Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Model File**: `ANOTHER_SOME_CHANGES_NEW_MODEL.keras`
- **Classes**: 29 (`A–Z`, `space`, `del`, `nothing`)

## Tech Stack

- Python
- TensorFlow
- OpenCV
- MediaPipe (`cvzone`)
- NumPy

## Usage

```bash
pip install -r requirements.txt
python main.py
