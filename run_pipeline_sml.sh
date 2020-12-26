#!/bin/bash
rm portfolios/sml_equity/data/*
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute MicroStrategy.ipynb
