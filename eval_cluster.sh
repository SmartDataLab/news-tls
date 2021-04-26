#!bash
DATASET=/data1/su/app/text_forecast/data/datasets_acl
RESULTS=/data1/su/app/text_forecast/results/acl
#python experiments/run_without_eval.py --dataset $DATASET --method clust --clust_ranker DateMention --summarizer Submodular --researcher su  --plug_page 0.2 --plug_taxo 2.5 --output $RESULTS/clust.json
python experiments/evaluate.py \
	--dataset $DATASET/nyt \
	--method clust \
	--output $RESULTS/nyt.clust.json
