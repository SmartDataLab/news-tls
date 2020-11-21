import argparse
from pathlib import Path
from news_tls import utils, data, datewise, clust, summarizers, plugin
from pprint import pprint
from pymongo import MongoClient
import time

CONN = MongoClient("localhost")
DB = CONN["news-tls"]
COLLECTION = DB["log"]


def stats_and_save(timeline, args, topic):
    taxo_distance = plugin.taxostat_distance(timeline.to_dict(), 4)
    print("taxo_distance", taxo_distance)
    taxo_purity = (
        sum(taxo_distance) / len(taxo_distance) if len(taxo_distance) > 0 else None
    )
    print("taxo_purity: %s" % taxo_purity)
    pages = plugin.get_timeline_pages(timeline.to_dict())
    print("pages: ", pages)
    headline_index = sum(pages) / len(pages) if len(pages) > 0 else None
    print("headline_index: %s" % headline_index)
    COLLECTION.insert_one(
        {
            "researcher": args.researcher,
            "topic": topic,
            "timeline": timeline.to_dict(),
            "record_time": time.time(),
            "taxo_distance": taxo_distance,
            "taxo_purity": taxo_purity,
            "pages": pages,
            "headline_index": headline_index,
            "method": args.method,
            "plug_page": args.plug_page,
            "plug_taxo": args.plug_taxo,
            "clust_ranker": args.clust_ranker,
            "date_ranker": args.date_ranker,
            "sent_collector": args.sent_collector,
            "summarizer": args.summarizer,
        }
    )


def run(tls_model, dataset, outpath, args):

    n_topics = len(dataset.collections)
    outputs = []

    for i, collection in enumerate(dataset.collections):
        topic = collection.name
        times = [a.time for a in collection.articles()]
        # setting start, end, L, K manually instead of from ground-truth
        collection.start = min(times)
        collection.end = max(times)
        l = 8  # timeline length (dates)
        k = 1  # number of sentences in each summary

        timeline = tls_model.predict(
            collection,
            max_dates=l,
            max_summary_sents=k,
        )

        print("*** TIMELINE ***")
        utils.print_tl(timeline)

        stats_and_save(timeline, args, topic)
        outputs.append(timeline.to_dict())

    if outpath:
        utils.write_json(outputs, outpath)


def main(args):

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    dataset = data.Dataset(dataset_path)
    dataset_name = dataset_path.name

    if args.method == "datewise":
        # load regression models for date ranking
        key_to_model = utils.load_pkl(args.model)
        models = list(key_to_model.values())
        if args.date_ranker == "Supervised":
            date_ranker = datewise.SupervisedDateRanker(method="regression")
            # there are multiple models (for cross-validation),
            # we just an arbitrary model, the first one
            date_ranker.model = models[0]
        else:
            date_ranker = DATE_RANKERS[args.date_ranker]()
        sent_collector = SENT_COLLECTORS[args.sent_collector]()
        summarizer = SUMMARIZERS[args.summarizer](plug=args.plug_page)
        system = datewise.DatewiseTimelineGenerator(
            date_ranker=date_ranker,
            summarizer=summarizer,
            sent_collector=sent_collector,
            key_to_model=key_to_model,
            plug_page=args.plug_page,
            plug_taxo=args.plug_taxo,
        )

    elif args.method == "clust":
        cluster_ranker = CLUST_RANKERS[args.clust_ranker]()
        clusterer = clust.TemporalMarkovClusterer()
        summarizer = SUMMARIZERS[args.summarizer](plug=args.plug_page)
        system = clust.ClusteringTimelineGenerator(
            cluster_ranker=cluster_ranker,
            clusterer=clusterer,
            summarizer=summarizer,
            clip_sents=2,
            unique_dates=True,
            plug_page=args.plug_page,
            plug_taxo=args.plug_taxo,
        )
    else:
        raise ValueError(f"Method not found: {args.method}")

    run(system, dataset, args.output, args)


CLUST_RANKERS = {
    "DateMention": clust.ClusterDateMentionCountRanker,
    "Size": clust.ClusterSizeRanker,
}

SUMMARIZERS = {
    "CentroidOpt": summarizers.CentroidOpt,
    "TextRank": summarizers.TextRank,
    "CentroidRank": summarizers.CentroidRank,
    "Submodular": summarizers.SubmodularSummarizer,
}

DATE_RANKERS = {
    "Random": datewise.RandomDateRanker,
    "MentionCount": datewise.MentionCountDateRanker,
    "PubCount": datewise.PubCountDateRanker,
    "Supervised": datewise.SupervisedDateRanker,
}

SENT_COLLECTORS = {
    "Mention": datewise.M_SentenceCollector,
    "Publish": datewise.P_SentenceCollector,
    "PM": datewise.PM_All_SentenceCollector,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--researcher", default="Anonymous", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--plug_page",
        default=False,
        type=float,
        help="0: do not use, x>0: weight of page rank",
    )
    parser.add_argument(
        "--plug_taxo",
        default=0,
        type=float,
        help="0: do not use, x>0: remove the node that with taxo-distance larger than x",
    )
    parser.add_argument(
        "--clust_ranker",
        default="DateMention",
        type=str,
        choices=["DateMention", "Size"],
        help="must be one of DateMention, Size",
    )
    parser.add_argument(
        "--date_ranker",
        default="Supervised",
        type=str,
        choices=["Random", "MentionCount", "PubCount", "Supervised"],
        help="must be one of Random, MentionCount, PubCount, Supervised",
    )
    parser.add_argument(
        "--sent_collector",
        default="PM",
        type=str,
        choices=["PM", "Mention", "Publish"],
        help="must be one of PM, Mention, Publish",
    )
    parser.add_argument(
        "--summarizer",
        default="CentroidOpt",
        type=str,
        choices=["CentroidOpt", "TextRank", "CentroidRank", "Submodular"],
        help="must be one of CentroidOpt, TextRank, CentroidRank, Submodular",
    )
    parser.add_argument("--model", default=None, help="model for date ranker")
    parser.add_argument("--output", default=None)
    main(parser.parse_args())
