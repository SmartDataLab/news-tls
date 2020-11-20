import pandas as pd


def get_page_sum_from_cluster(cluster):
    return sum([int(a.page) for a in cluster.articles])


def get_combined_1st_rank(count_dict, page_weight=1.0, output_one=True) -> str:
    """
    input: {'date':(count,page_sum)}
    output: 'date_with_most_influence'
    """
    rank_dict = {}
    min_rank = 1000
    count_sort = sorted(list(count_dict.items()), key=lambda x: x[1][0], reverse=True)
    page_sort = sorted(list(count_dict.items()), key=lambda x: x[1][1])
    for i in range(len(count_sort)):
        rank_dict[count_sort[i][0]] = i + 1

    for i in range(len(page_sort)):
        value = (i + 1) * page_weight + rank_dict[page_sort[i][0]]
        if output_one:
            if value < min_rank:
                min_rank = value
                rank_1st = page_sort[i][0]
        else:
            rank_dict[page_sort[i][0]] = value
    if output_one:
        return rank_1st
    else:
        return sorted(list(rank_dict.items()), key=lambda x: x[1])


def taxostat_distance(timeline, depth) -> list:
    """
    params:
    timeline: 一条时间线，包括日期与文本
    [['date0', ['id : raw_taxostr : page : text']],
    ['date1', ['id : raw_taxostr : page : text'],
    ...]
    depth: taxonomic classifier的最大追索深度,一般最大距离为depth-1

    return: list,每一个时间节点距离基准taxonomic classifier的平均距离
    """
    # 取出raw_taxostr
    raw_taxostr_lst = []

    for curr_day in timeline:
        raw_taxostr_lst.append(curr_day[1][0].split(" : ")[1])
    # 划分taxostr
    taxostr_lst = [raw_taxostr.split("|") for raw_taxostr in raw_taxostr_lst]
    # taxostr分段
    taxo_unit_lst = [[taxostr.split("/") for taxostr in unit] for unit in taxostr_lst]

    # 计算距离
    # 以出现频次最高的taxo为基准
    base_taxo = (
        pd.value_counts([taxostr for unit in taxostr_lst for taxostr in unit])
        .index[0]
        .split("/")
    )
    base_len = len(base_taxo)
    # 计算每个时间节点内taxo的平均距离
    taxo_distance_lst = []
    for taxo_unit in taxo_unit_lst:
        curr_scores = []
        for taxo in taxo_unit:  # 计算每一个taxo距离base_taxo的距离
            minus = 1
            for i in range(min(base_len, len(taxo))):
                if taxo[i] != base_taxo[i]:
                    minus = 0
                    break

            score = depth - minus - i
            curr_scores.append(score)

        taxo_distance_lst.append(sum(curr_scores) * 1.0 / len(taxo_unit))

    return taxo_distance_lst


def get_timeline_pages(timeline):
    return [float(curr_day[1][0].split(" : ")[2]) for curr_day in timeline]