#!/bin/bash
rm data/*
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute DownloadData.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1  --to notebook --execute AddIndicators.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1  --to notebook --execute Strategy.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1  --to notebook --execute GenerateSignalsMR.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1  --to notebook --execute GenerateSignalsTrend.ipynb

