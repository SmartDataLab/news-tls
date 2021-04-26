#!bash
DATASET=/data1/su/app/text_forecast/hxw/data/test
HEIDELTIME=/data1/su/app/text_forecast/tilse-master/tilse/tools/heideltime
python preprocess_tokenize.py --dataset $DATASET
python preprocess_heideltime.py --dataset $DATASET --heideltime $HEIDELTIME
python preprocess_spacy.py --dataset $DATASET
