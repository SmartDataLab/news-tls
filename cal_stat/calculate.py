import numpy as np
import pandas as pd
import json
import pickle
from TaxoTree import *
import argparse

# 匹配raw_taxostr
def match_raw_taxostr(sota_result, datafile):
    # 获取文章id
    article_ids = [text.split(' : ')[0] for item in sota_result for text in item[1]]
    
    # 获得year-identification字典
    year_identification_dict = {}
    for idx in article_ids:
        year = idx[0:4]
        identification = tuple(idx.split('-'))
        if year_identification_dict.get(year, None) is None:
            year_identification_dict[year] = [identification]
        else:
            year_identification_dict[year].append(identification)
            
    # 遍历字典，匹配日期与Slug
    raw_taxostr_lst = []
    for year in year_identification_dict.keys():
        df = pd.read_csv(datafile+'/'+year+'.csv')[['Publication Date', 'Slug', 'Taxonomic Classifiers']]
        df['Publication Date'] = df['Publication Date'].apply(lambda x: x[0:8])  
        
        # 匹配
        curr_raw_taxostr_lst = []
        identification_lst = year_identification_dict[year]
        for identification in identification_lst:
            raw_taxostr = df[(df['Publication Date'] == identification[0]) * (df['Slug'] == identification[1])]['Taxonomic Classifiers']
            curr_raw_taxostr_lst.extend(raw_taxostr)
        
        raw_taxostr_lst.extend(curr_raw_taxostr_lst)
    
    return raw_taxostr_lst

# 对taxo去重并整理格式
def generate_taxo_unit_lst(raw_taxo = None, filename = None, colname = 'Taxonomic Classifiers'):
    """
    raw_taxo: pd.Series([raw_taxostr, raw_taxotr, ...])
    output: [[[label00, label01], [label10, label11, label12]], []
             ...]
    """
    
    if raw_taxo is None:
        # 读入数据
        df = pd.read_csv(filename)
        print('Data readin.')
        # 取出taxostr
        taxostr = df[colname].apply(lambda x: str(x).split('|'))
    else:
        # 将raw taxostr按照|切分
        taxostr = pd.Series(raw_taxo).apply(lambda x: str(x).split('|'))
    
    print('Taxonomic Classifiers extracted.')
    
    # 按照字符创长短排序
    taxostr = [sorted(taxostr_unit, key = lambda x: len(x), reverse = True)
               for taxostr_unit in taxostr]
    # 对于每一篇文章，保留更精细的taxo
    taxostr_simple = []
    for taxostr_unit in taxostr:
        curr_set = []  # 为每一篇文章生成不重复的taxo集合
        for taxostr in taxostr_unit:  # 检查当前taxostr是否为父级分类
            add = True
            strlen = len(taxostr)
            for former_taxostr in curr_set:
                if former_taxostr[0:strlen] == taxostr:
                    add = False
                    break
            if add:  # 不是父级分类，加入集合
                curr_set.append(taxostr)
            
        taxostr_simple.append(curr_set) 
    print('Taxo simplified.')
        
    # 将字符串分割为多级label
    taxo_unit_lst = [[taxo.split('/') for taxo in taxo_unit] for taxo_unit in taxostr_simple]
    print('Taxo split.')
    
    return taxo_unit_lst

def search_all_routes_by_depth(taxo_unit_lst, depth, taxotree = None):
    """
    return: {taxostr0: [id0, id1, ...],
             taxostr1: [id0, id1, ...],
             ...}
    """
    if taxotree is None:
        # 对每一篇文章，获取一定深度的taxostr集合
        taxostr_depth_lst = [set(['/'.join(taxo[0:depth]) if len(taxo) >= depth else '/'.join(taxo) for taxo in taxo_unit])
                             for taxo_unit in taxo_unit_lst]
        
        taxo_dict = {}
        # 遍历每一篇文章的id与taxostr_set
        for idx, taxostr_set in enumerate(taxostr_depth_lst):
            for taxostr in taxostr_set:
                if taxo_dict.get(taxostr, None) is None:
                    taxo_dict[taxostr] = [idx]
                else:
                    taxo_dict[taxostr].append(idx)
        
    else:
        depth = min(taxotree.level, depth)
        #获得一定深度下的不重复taxostr集合
        taxostr_depth_set = set(['/'.join(taxo[0:depth]) if len(taxo) >= depth else '/'.join(taxo) 
                                 for taxo_unit in taxo_unit_lst for taxo in taxo_unit])
        taxo_dict = {}
        for taxostr in taxostr_depth_set:
            taxo = taxostr.split('/')
            taxo_dict[taxostr] = taxotree.get_articles(taxo)
        
    
    return taxo_dict

def taxostat_percent(taxo_dict, taxotree, args):
    total = args[0]
    taxo_percent_dict = {}
    for taxostr, ids in taxo_dict.items():
        taxo_percent_dict[taxostr] = len(ids)*1.0/total if total > 0 else None
        
    return taxo_percent_dict

def taxostat_similarity(taxo_dict, taxotree = None, args = [3]):
    """
    args: [depth, mean]
    """
    depth = args[0]

    # 找到基准taxo
    sorted_dict = sorted(taxo_dict.items(), key = lambda x: len(x[1]), reverse = True)
    for item in sorted_dict:
        taxo = item[0].split('/')
        if len(taxo) == depth:
            break
    base_taxo = taxo

    score_dict = {}
    for taxostr, id_lst in sorted_dict:
        # 计算得分
        taxo = taxostr.split('/')
        minus = 0
        for i in range(min(len(taxo), depth)):
            if taxo[i] != base_taxo[i]:
                minus = 1
                break
        score = i - minus
        # 记录得分
        for idx in id_lst:
            if score_dict.get(idx, None) is None:
                score_dict[idx] = [score]
            else:
                score_dict[idx].append(score)

    # 计算每篇文章的最终得分
    final_score_lst = []
    for score_lst in score_dict.values():
        final_score_lst.append(np.mean(score_lst))

    # 输出聚类结果的平均得分
    return np.mean(final_score_lst)


# 所有计算统计量的函数的字典
stat_func_dict = {
    'percent': taxostat_percent,
    'similarity': taxostat_similarity
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonfile', required=True)
    parser.add_argument('--datafile', required=True)
    parser.add_argument('--taxodepth', default = 3, type = int)
    parser.add_argument('--store', required=True)
    parser.add_argument('--stat', default='percent')
    
    args = parser.parse_args()
    
    if args.stat not in stat_func_dict.keys():
        print('Invalid Statistics!')
        return None
    
    # 读入数据
    all_sota_results = json.load(open(args.jsonfile, 'r'))
    print('Successfully load {} sets of sota results'.format(len(all_sota_results)))
    
    all_stat = {}
    for idx, sota_res in enumerate(all_sota_results):
        raw_taxostr_lst = match_raw_taxostr(sota_res, args.datafile)
        taxo_unit_lst = generate_taxo_unit_lst(raw_taxostr_lst)
        
        taxotree = TaxoTree(taxo_unit_lst, by_article = True)
        taxo_dict = search_all_routes_by_depth(taxo_unit_lst, args.taxodepth)
        
        if args.stat == 'percent':
            add_args = [len(taxo_unit_lst)]
        elif args.stat == 'similarity':
            add_args = [args.taxodepth]
        curr_stat = stat_func_dict[args.stat](taxo_dict, taxotree, add_args)
        # print(curr_stat)

        all_stat[str(idx)] = curr_stat
        
    json.dump(all_stat, open(args.store, 'w'))
    

if __name__ == '__main__':
    main()
