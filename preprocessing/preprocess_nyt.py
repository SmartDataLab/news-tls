import jsonlines
import json
import pandas as pd
import re
import argparse
import arrow
from tqdm import tqdm
import os

NYT_DATA_PATH = "../../../data/nyt_csv"


def get_clean_words_set_from_abstract(abstract):
    text = abstract.lower()
    # foction de replacement
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}=~|.?,]", "", text)
    return text


def main(args):
    keyword = args.keywords.split(",")
    keywords = set(keyword)
    topic_name = args.keywords.replace(" ", "_")
    os.mkdir(args.dataset_dir + "/%s" % topic_name)
    with open(args.dataset_dir + "/%s/keywords.json" % topic_name, "w") as f:
        f.write(json.dumps(keyword))
    file = jsonlines.open(args.dataset_dir + "/%s/articles.jsonl" % topic_name, "w")
    for year in tqdm(range(args.start_year, args.end_year)):
        df = pd.read_csv(NYT_DATA_PATH + "/%s.csv" % year, low_memory=False)
        for row in df[
            [
                "Publication Date",
                "Slug",
                "Article Abstract",
                "Page",
                "Taxonomic Classifiers",
            ]
        ].iterrows():
            try:
                if args.mode == "or":
                    query = not keywords.isdisjoint(
                        get_clean_words_set_from_abstract(
                            row[1]["Article Abstract"]
                        ).split(" ")
                    )
                elif args.mode == "and":
                    query = keywords.issubset(
                        get_clean_words_set_from_abstract(
                            row[1]["Article Abstract"]
                        ).split(" ")
                    )
            except:
                query = False
            # print(type(row))
            if query:
                jsonlines.Writer.write(
                    file,
                    {
                        "id": row[1]["Publication Date"].split("T")[0]
                        + "-"
                        + row[1]["Slug"],
                        "time": row[1]["Publication Date"],
                        "text": row[1]["Article Abstract"],
                        "taxo": row[1]["Taxonomic Classifiers"],
                        "page": row[1]["Page"],
                    },
                )
    file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, help="dataset directory")
    parser.add_argument(
        "--keywords", required=True, help="for example 'elon,elon musk,musk' "
    )
    parser.add_argument("--start_year", default=2006, type=int)
    parser.add_argument("--end_year", default=2008, type=int)
    parser.add_argument("--mode", default="or", choices=["or", "and"])
    main(parser.parse_args())