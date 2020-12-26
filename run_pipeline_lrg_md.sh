#!/bin/bash
rm portfolios/lrg_med_equity/data/*
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute LargeAndMidStrategy.ipynb