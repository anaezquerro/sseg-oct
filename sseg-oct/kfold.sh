#!/bin/bash

# kwargs

models="base unet linknet pspnet pan segmenter attnunet deformunet"
kwargs="$@"


for model in $models
do
  python3 test.py $model kfold --verbose $kwargs
done

