del /Q data\*
jupyter nbconvert --ExecutePreprocessor.timeout=-1  --to notebook --execute DownloadData.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1  --to notebook --execute AddIndicators.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1  --to notebook --execute Strategy.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1  --to notebook --execute GenerateSignals.ipynb