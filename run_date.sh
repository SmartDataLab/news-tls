#!bash
DATASET=/data1/su/app/text_forecast/hxw/data/test
RESULTS=/data1/su/app/text_forecast/hxw/data/sota_res
python experiments/run_without_eval.py --dataset $DATASET --method datewise --model resources/datewise/supervised_date_ranker.entities.pkl --output $RESULTS/datewise.json
