#!/usr/bin/env bash

mkdir -p data
cd data
kaggle datasets download evanarlian/eedi-paraphrased
unzip eedi-paraphrased.zip -d eedi-paraphrased
