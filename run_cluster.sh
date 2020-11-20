#!bash
DATASET=/data1/su/app/text_forecast/hxw/data/case
RESULTS=/data1/su/app/text_forecast/results/test
python experiments/run_without_eval.py --dataset $DATASET --method clust --clust_ranker DateMention --summarizer Submodular --researcher su  --plug_page 0.2 --plug_taxo 2.5 --output $RESULTS/clust.json
