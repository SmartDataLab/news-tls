#!bash
DATASET=/data1/su/app/text_forecast/data/datasets/test
RESULTS=/data1/su/app/text_forecast/results/test
python experiments/run_without_eval.py --dataset $DATASET --plug_page 0.2 --plug_taxo 2.5 --method datewise --model resources/datewise/supervised_date_ranker.entities.pkl --summarizer Submodular --date_ranker Supervised --sent_collector Publish --researcher su --output $RESULTS/datewise.json
