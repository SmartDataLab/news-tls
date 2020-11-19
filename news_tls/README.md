# Methodology

## mainstream methods intro
- direct summary

- datewise summary
    + date selection
        - pubcount
        - mentioncount(2nd)
        - supervised(1st)
        - Ours: metioncount with page weight and calculate its taxonomic purity
    + candidate sentences for dates
        - published or closely after
        - mention
        - PM-MEAN
        - Ours: PM-MEAN with influence-rank
    + date summaries
        - textrank
        - centroid-rank
        - centroid-opt
        - submodular
        - Ours: centroid-opt with influence-rank

- event detection
    + clustering(Markov Clustering)
        - MCL is a clustering algorithm, it is based on simulation random walks along nodes in a graph
    + assigning date to clusters(influence-based date rank)
    + cluster ranking
        - size
        - datemention count(1st)
        - regression
        - Ours
    + cluster summarization
        - Centroid-Opt
        - Ours: Centroid-Opt with influence ranking

- Our ituition
    + frequency-based summary -> influence-rank summary(there is a hyperparameter factor)
    + naive timeline -> timeline distillation module