#!/bin/bash -x

kaggle competitions download -p input/ -c g2net-gravitational-wave-detection
kaggle datasets     download -p input/ -d yasufuminakama/g2net-n-mels-128-train-images
kaggle datasets     download -p input/ -d yasufuminakama/g2net-n-mels-128-test-images

cd input/
for FILE in *.zip; do
  DIRNAME=`basename $FILE .zip`
  mkdir $DIRNAME;
  unzip -n $FILE -d $DIRNAME;
done;
