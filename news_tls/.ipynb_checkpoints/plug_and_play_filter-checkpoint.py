import pandas as pd


def get_combined_1st_rank(count_dict) -> str:
    """
    input: {'date':(count,page_sum)}
    output: 'date_with_most_influence'
    """
    sorted(list(count_dict.values()), key=lambda x: x[1][1])
