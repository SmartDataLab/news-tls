# Calculate statistics from sota results

## General Description

We provide a series of functions to match raw taxonomic strings based on the publication date and slug, split taxonomic classifiers, and eliminate repeated classifiers with certain depth.

We also provide a tree-structured class `TaxoTree` to record the hierarchical structure of taxonomic classifiers, which is specified in `TaxoTree.py`.

The functions to calculate statistics from taxonomic classifiers have the identical prefix `taxostat_`. Currently we can calculate the following statistics:

* Percent of each taxonomic classifier in the time-line; 
* Score of similarity of the time-line;

## Code

This is an example to calculate percent of each classifier with depth 4:

```bash
python calculate.py --jsonfile $SOTA_RES/clust.json \
                    --datafile $DATA/nyt_csv \
                    --taxodepth 4 \
                    --stat percent \
                    --store $STORE/percent.json
```

This is an example to calculate the similarity of classifiers in the time-line with depth 4:

```bash
python calculate.py --jsonfile $SOTA_RES/clust.json \
                    --datafile $DATA/nyt_csv \
                    --taxodepth 4 \
                    --stat similarity \
                    --store $STORE/similarity.json 
```

