#!/usr/bin/env bash

mkdir -p data
cd data
kaggle datasets download evanarlian/eedi-synthetic
unzip eedi-synthetic.zip -d eedi-synthetic
