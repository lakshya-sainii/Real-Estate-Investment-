#!/bin/sh
# Full pipeline: preprocess -> EDA -> classification -> regression -> app artifacts
cd "$(dirname "$0")/src" || exit 1
python3 preprocess.py && python3 eda.py && python3 train_classification.py && python3 train_regression.py && python3 export_portable_classifier.py && python3 build_app_data.py
