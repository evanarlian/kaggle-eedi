#!/usr/bin/env bash

mkdir -p data
cd data
kaggle competitions download -c eedi-mining-misconceptions-in-mathematics
unzip eedi-mining-misconceptions-in-mathematics.zip
