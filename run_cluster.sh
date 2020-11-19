#!bash
DATASET=/data1/su/app/text_forecast/data/datasets/test
RESULTS=/data1/su/app/text_forecast/results/test
python experiments/run_without_eval.py --dataset $DATASET --method clust --clust_ranker Size --summarizer TextRank --plug_page 0.2 --plug_taxo 2.5 --output $RESULTS/clust.json
