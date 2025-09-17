# Disease Classifier

![Python](https://img.shields.io/badge/python-3.8%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)

A simple, modular Python project that trains a disease classification model on a labeled dataset and predicts disease from user input (symptoms / features). Designed for experimentation and easy productionization (model saving, REST API, containerization).

---

## Table of contents
- [Overview](#overview)
- [Features](#features)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Quickstart](#quickstart)
  - [Train the model](#train-the-model)
  - [Evaluate the model](#evaluate-the-model)
  - [Predict (CLI)](#predict-cli)
  - [Serve (FastAPI)](#serve-fastapi)
- [Example output](#example-output)
- [Docker (optional)](#docker-optional)
- [Tips to improve](#tips-to-improve)
- [Contributing](#contributing)
- [License & Acknowledgements](#license--acknowledgements)

---

## Overview
This project trains a classification model using a CSV dataset (features + target). The trained pipeline (preprocessing + classifier) is saved and can be used to:

- predict disease from single or batch inputs (CLI / JSON)
- serve predictions via a FastAPI endpoint
- inspect model performance (accuracy, F1, confusion matrix)

The code is intentionally modular so you can swap datasets, models, or preprocessing steps easily.

---

## Features
- Training pipeline with model persistence (`joblib` / `pickle`)
- CLI prediction script for single-sample inference
- Evaluation utilities (accuracy, precision, recall, F1, confusion matrix)
- Optional FastAPI app to serve predictions
- Example Dockerfile for containerized deployment
- Guidelines for model interpretability and improvements

---

