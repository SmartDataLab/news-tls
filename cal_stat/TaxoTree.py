import numpy as np
import pandas as pd


class TaxoNode:
    def __init__(self, label, children = None, contents = None):
        """
        label: 标签，str
        children: 子节点列表，format: [child1, ..., childm]
        contents: 终止于当前节点的个体标识列表，format: [id1, id2, ..., idn]
        """
        self.label = label
        self.children = None
        self.contents = None  # set(), 不重复
        
        # children
        # format: {'label1': child1, ..., 'labelm': childm}
        if children is not None:
            for child in children:
                self.add_child(child)
                
        # contents
        if contents is not None:
            self.contents = set()
            for item in contents:
                self.contents.add(item)
    
    def print(self):
        print('Children:', self.children.keys() if self.children is not None else None)
        print('Contents:', list(self.contents) if self.contents is not None else None)
    
    # 添加个体标识
    # 重复不提醒，但也不重复记录
    def record(self, content):
        if self.contents is None:
            self.contents = set([content])
        else:
            self.contents.add(content)
    
    # 添加子节点
    def add_child(self, child):
        if self.children is None:
            self.children = {child.label: child}
        else:
            self.children[child.label] = child 
    
    # 标识是否存在       
    def is_content(self, content):
        if self.contents is None:
            return False
        else:
            return content in self.contents
     
    # 标签是否在子节点中
    def is_child_label(self, label):
        if self.children is None:
            return False
        else:
            return label in self.children.keys()
    
    # 子节点数量
    def children_num(self):
        if self.children is None:
            return 0
        else:
            return len(self.children.keys())
      
    # 标识数量
    def contents_num(self):
        if self.contents is None:
            return 0
        else:
            return len(self.contents)
        
    # 挪动至子节点
    def to_child(self, label):
        return self.children.get(label, None)
    
    # 获得内容
    def get_contents(self):
        if self.contents is None:
            return None
        else:
            return [item for item in self.contents]

class TaxoTree:
    def __init__(self, taxo_lst, by_article = False, article_lst = None):
        self.root = None
        self.level_labels = None
        self.level = 0
        self.node_num = 0
        
        if taxo_lst is not None:
            # 仅仅记录分类
            # taxo_lst: [taxo1, taxo2,..., taxon]
            if not by_article:
                for taxo in taxo_lst:
                    self.record_taxo(taxo)
                    
            # 记录分类且添加对应文章
            # taxo_lst: [[taxo11, taxo12,..., taxo1m1],..., [taxon1, taxon2,..., taxomn]
            else:
                if article_lst is None:  # 生成编号
                    article_lst = [i for i in range(len(taxo_lst))]
                 
                if len(article_lst) != len(taxo_lst):  # 文章编号与taxo_lst长度不匹配
                    print('The list of article ids is invalid!')
                    return None
                
                for i in range(len(article_lst)):
                    article = article_lst[i]  # 文章标识
                    taxo_unit = taxo_lst[i]  # 文章的全部taxo
                    for taxo in taxo_unit:
                        self.record_taxo(taxo)  # 记录分类
                        self.add_article(article, taxo)  # 添加文章

    # 记录的文章总数
    def num_article(self):
        if self.root is None:
            return 0
        else:
            return self.root.contents_num()
    
    # 每一层的标签数
    def num_per_layer(self):
        if self.root is None:
            return []
        else:
            return [len(layer) for layer in self.level_labels]
        
    # 打印指定层标签
    def print_layer(self, layer):
        if self.root is None:
            print('Empty')
            return
        
        if layer > self.level - 1:
            print('Invalid layer!')
            return
        
        print('Layer', layer)
        print(self.level_labels[layer])
        
    
    # 打印TaxoTree相关信息
    def print(self, layer_nums = None):
        if self.root is None:
            print('Empty')
            return
        
        if layer_nums is None:
            layer_nums = self.level
        print("Number of layers:", self.level)
        print("Number of nodes:", self.node_num)
        print("Number of articles recorded:", self.num_article())
        print("======Labels======")
        print_layers = min(self.level, layer_nums)
        for i in range(print_layers):
            print("Layer", i, self.level_labels[i])
     
    # 返回指定分类中的标识
    def get_articles(self, taxo):
        if self.root is None:
            print('The TaxoTree is empty!')
            return None
        
        if self.root.label != taxo[0]:
            print('The taxo is invalid!')
            return None
        
        tree = self.root
        for label in taxo[1:]:
            if tree.is_child_label(label):
                tree = tree.to_child(label)
            else:
                print('The taxo is invalid!')
                return None
            
        return tree.get_contents()
                
    # 将文章标识加入taxo_tree
    def add_article(self, article_id, taxo, invalid = False):
        if self.root is None and invalid:
            print('The TaxoTree is empty!')
            return None
        
        if self.root.label != taxo[0] and invalid:
            print('The taxo is invalid!')
            return None
        
        tree = self.root
        tree.record(article_id)
        for label in taxo[1:]:
            if tree.is_child_label(label):  # 下一阶段匹配成功
                tree = tree.to_child(label)
                tree.record(article_id)
            else:  # 下一阶段匹配不成功
                if invalid:
                    print('Stop Record!')
                return
   
    # 添加路径
    def record_taxo(self, taxo, exist = False, invalid = False):
        # 初始化
        if self.root is None:
            self.root = TaxoNode(taxo[0])
            self.level_labels = [[taxo[0]]]
            self.level = 1
            self.node_num = 1
        
        # 检查根节点是否匹配
        if taxo[0] != self.root.label and invalid:
            print('Invalid Input!')
            return None
        
        # 检查并添加路径
        # 添加节点   
        flag = True  # 是否已经存在
        taxo_len = len(taxo)
        curr_tree = self.root
        for i in range(1, taxo_len, 1):
            label = taxo[i]
            
            if not curr_tree.is_child_label(label):
                flag = False
                new_node = TaxoNode(label)
                curr_tree.add_child(new_node)
                # 添加label
                if len(self.level_labels) < i+1:
                    self.level_labels.append([label])
                else:
                    self.level_labels[i].append(label)
                # 计数
                self.node_num += 1
            
            curr_tree = curr_tree.to_child(label)
        
        # 修改层数
        if taxo_len > self.level:
            self.level = taxo_len
            
        if exist:
            if flag:
                print('The taxo has already existed.')
            else:
                print('New taxo recorded!')
            
        # if self.level == len(self.level_labels):
        #     print('Success!')

