import numpy as np
import datetime
import itertools
import random
import collections
import markov_clustering as mc
from sklearn.feature_extraction.text import TfidfVectorizer#用来提取文本特征的
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse  #sklearn.feature_extraction.text方法处理后的数据格式是scipy.sparse的，是稀疏矩阵
from typing import List
from news_tls import utils, data, plugin

#生成时间线
class ClusteringTimelineGenerator:
    def __init__(
        self,
        clusterer=None,
        cluster_ranker=None,
        summarizer=None,
        clip_sents=5,
        key_to_model=None,
        unique_dates=True,
        plug_page=False,
        plug_taxo=False,
    ):
        self.plug_page = plug_page
        self.plug_taxo = plug_taxo
        self.clusterer = clusterer or TemporalMarkovClusterer()
        self.cluster_ranker = cluster_ranker or ClusterDateMentionCountRanker()
        self.summarizer = summarizer or summarizers.CentroidOpt(plug=self.plug_page)
        self.key_to_model = key_to_model
        self.unique_dates = unique_dates
        self.clip_sents = clip_sents

    def predict(
        self,
        collection,
        max_dates=10,
        max_summary_sents=1,
        ref_tl=None,
        input_titles=False,
        output_titles=False,
        output_body_sents=True,
    ):

        print("clustering articles...")#正在聚类文章……
        doc_vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")#这是定义了一个方法？
        clusters = self.clusterer.cluster(collection, doc_vectorizer)#对对象进行聚类

        print("assigning cluster times...")#正在聚类时间……
        for c in clusters:
            c.time = c.most_mentioned_time()#把时间定义为最常被提起的时间
            if c.time is None:
                c.time = c.earliest_pub_time()#如果没有最常被提起的时间：就定义为最早发布的时间

        print("ranking clusters...")#正在对聚类结果进行排序……
        ranked_clusters = self.cluster_ranker.rank(
            clusters, collection, plug=self.plug_page
        )#讲道理有点看不懂，因为不知道数据是什么格式的

        print("vectorizing sentences...")#正在向量化句子（embedding?）
        raw_sents = [
            s.raw for a in collection.articles() for s in a.sentences[: self.clip_sents]
        ]#我甚至没看到过这种语法，连续两个for是并列循环还是嵌套循环？#知道了，是嵌套循环，甚至是按语法的前后顺序循环的
        vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")#向量编码器
        vectorizer.fit(raw_sents)#用raw_sents的方法还是数据去训练（拟合）向量编码器

        def sent_filter(sent):#在“预测”方法下面继续定义一个方法“传送过滤”？
            """
            Returns True if sentence is allowed to be in a summary.
            """#就是说如果这个句子被允许存在于摘要中就返回“True”。
            lower = sent.raw.lower()#全变成小写字母，但是.raw是什么意思我看不懂？
            if not any([kw in lower for kw in collection.keywords]):#如果一个都没有，就返回False
                return False
            elif not output_titles and sent.is_title:#如果？？并且？？，就返回False
                return False
            elif not output_body_sents and not sent.is_sent:#如果？？并且？？，就返回False
                return False
            else:#剩余情况返回True
                return True
            #总的来说这个方法是用来根据sent信息进行筛选的，但是不知道是用来筛选什么的。

        print("summarization...")#摘要……
        sys_l = 0#生成两个不知道干嘛用的初始值
        sys_m = 0
        ref_m = max_dates * max_summary_sents

        date_to_summary = collections.defaultdict(list)#日期到摘要：

        for c in ranked_clusters:#一个循环：对排好序的聚类结果进行循环

            date = c.time.date()#这样我们知道了之前的聚类结果应该有一个属性叫做time.date()
            c_sents = self._select_sents_from_cluster(c)#从聚类选择传送

            summary = self.summarizer.summarize(
                c_sents, k=max_summary_sents, vectorizer=vectorizer, filter=sent_filter
            )#摘要为根据数据格式产生的方法，根据下文来看，返回的是True/False结果。#但是根据下面if后面第二行来看，summary反而是一个可迭代的对象？#破案了，只要不确定为0或者空列表/元组，作为布尔值就是True（哪怕全是0的列表或者元组也认为是True）。

            if summary:#如果确定摘要里要加这一部分：#那么这里summary应该是一个有可能为0但一般情况下不为空的一个
                c_sents_raw = [s.raw for s in c_sents]#如果这里的raw也是“未经加工过”的意思：那么这句话就是把之前的c_sents列表用raw方法重构一下
                idx = c_sents_raw.index(summary[0])#list.index()：#用于找到值等于给定值的第一个匹配项的索引。
                if self.unique_dates and date in date_to_summary:
                    continue
                date_to_summary[date] += [
                    "%s : %s : %s : "
                    % (
                        c_sents[idx].article_id,
                        c_sents[idx].article_taxo,
                        c_sents[idx].article_page,
                    )
                    + summary[0]
                ]
                sys_m += len(summary)
                if self.unique_dates:
                    sys_l += 1

            if sys_m >= ref_m or sys_l >= max_dates:
                break

        timeline = []
        for d, summary in date_to_summary.items():
            t = datetime.datetime(d.year, d.month, d.day)
            timeline.append((t, summary))
        timeline.sort(key=lambda x: x[0])
        if self.plug_taxo:
            distances = plugin.taxostat_distance(timeline, 4)
            timeline = [
                timeline[i]
                for i, dist in enumerate(distances)
                if dist <= self.plug_taxo
            ]

        return data.Timeline(timeline)

    def _select_sents_from_cluster(self, cluster):
        sents = []
        for a in cluster.articles:
            pub_d = a.time.date()
            for s in a.sentences[: self.clip_sents]:
                sents.append(s)

        return sents

    def load(self, ignored_topics):
        pass

#聚类
################################# CLUSTERING ###################################


class Cluster:
    def __init__(self, articles, vectors, centroid, time=None, id=None):
        self.articles = sorted(articles, key=lambda x: x.time)
        self.centroid = centroid
        self.id = id
        self.vectors = vectors
        self.time = time

    def __len__(self):
        return len(self.articles)

    def pub_times(self):
        return [a.time for a in self.articles]

    def earliest_pub_time(self):
        return min(self.pub_times())

    def most_mentioned_time(self):
        mentioned_times = []
        for a in self.articles:
            for s in a.sentences:
                if s.time and s.time_level == "d":
                    mentioned_times.append(s.time)
        if mentioned_times:
            return collections.Counter(mentioned_times).most_common()[0][0]
        else:
            return None

    def most_mentioned_time_with_influence_rank(self, page_weight=1.0):
        count_dict = {}

        for a in self.articles:
            for s in a.sentences:
                if s.time and s.time_level == "d":
                    count, page_sum = count_dict.get(s.time, (0, 0))
                    count_dict[s.time] = (count + 1, page_sum + int(s.article_page))
        count_dict = {
            key: (tuple_[0], tuple_[1] / tuple_[0])
            for key, tuple_ in count_dict.items()
        }
        if count_dict:  # equals to "if len(mentioned_times) != 0:"
            return plugin.get_combined_1st_rank(count_dict, page_weight)
        else:
            return None

    def update_centroid(self):
        X = sparse.vstack(self.vectors)
        self.centroid = sparse.csr_matrix.mean(X, axis=0)


class Clusterer:
    def cluster(self, collection, vectorizer) -> List[Cluster]:
        raise NotImplementedError


class OnlineClusterer(Clusterer):
    def __init__(self, max_days=1, min_sim=0.5):
        self.max_days = max_days
        self.min_sim = min_sim

    def cluster(self, collection, vectorizer) -> List[Cluster]:
        # build article vectors
        texts = ["{} {}".format(a.title, a.text) for a in collection.articles]
        try:
            X = vectorizer.transform(texts)
        except:
            X = vectorizer.fit_transform(texts)

        id_to_vector = {}
        for a, x in zip(collection.articles(), X):
            id_to_vector[a.id] = x

        online_clusters = []

        for t, articles in collection.time_batches():
            for a in articles:
                # calculate similarity between article and all clusters
                x = id_to_vector[a.id]
                cluster_sims = []
                for c in online_clusters:
                    if utils.days_between(c.time, t) <= self.max_days:
                        centroid = c.centroid
                        sim = cosine_similarity(centroid, x)[0, 0]
                        cluster_sims.append(sim)
                    else:
                        cluster_sims.append(0)

                # assign article to most similar cluster (if over threshold)
                cluster_found = False
                if len(online_clusters) > 0:
                    i = np.argmax(cluster_sims)
                    if cluster_sims[i] >= self.min_sim:
                        c = online_clusters[i]
                        c.vectors.append(x)
                        c.articles.append(a)
                        c.update_centroid()
                        c.time = t
                        online_clusters[i] = c
                        cluster_found = True

                # initialize new cluster if no cluster was similar enough
                if not cluster_found:
                    new_cluster = Cluster([a], [x], x, t)
                    online_clusters.append(new_cluster)

        clusters = []
        for c in online_clusters:
            cluster = Cluster(c.articles, c.vectors)
            clusters.append(cluster)

        return clusters

#但是我没找到聚类中心在哪里
class TemporalMarkovClusterer(Clusterer):
    def __init__(self, max_days=1):
        self.max_days = max_days

    def cluster(self, collection, vectorizer) -> List[Cluster]:
        articles = list(collection.articles())
        texts = ["{} {}".format(a.title, a.text) for a in articles]
        try:
            X = vectorizer.transform(texts)
        except:
            X = vectorizer.fit_transform(texts)

        times = [a.time for a in articles]

        print("temporal graph...")
        S = self.temporal_graph(X, times)
        # print('S shape:', S.shape)
        print("run markov clustering...")
        result = mc.run_mcl(S)
        print("done")

        idx_clusters = mc.get_clusters(result)
        idx_clusters.sort(key=lambda c: len(c), reverse=True)

        print(
            f"times: {len(set(times))} articles: {len(articles)} "
            f"clusters: {len(idx_clusters)}"
        )

        clusters = []
        for c in idx_clusters:
            c_vectors = [X[i] for i in c]
            c_articles = [articles[i] for i in c]
            Xc = sparse.vstack(c_vectors)
            centroid = sparse.csr_matrix(Xc.mean(axis=0))
            cluster = Cluster(c_articles, c_vectors, centroid=centroid)
            clusters.append(cluster)

        return clusters

    def temporal_graph(self, X, times):
        times = [utils.strip_to_date(t) for t in times]
        time_to_ixs = collections.defaultdict(list)
        for i in range(len(times)):
            time_to_ixs[times[i]].append(i)

        n_items = X.shape[0]
        S = sparse.lil_matrix((n_items, n_items))
        start, end = min(times), max(times)
        total_days = (end - start).days + 1

        for n in range(total_days + 1):
            t = start + datetime.timedelta(days=n)
            window_size = min(self.max_days + 1, total_days + 1 - n)
            window = [t + datetime.timedelta(days=k) for k in range(window_size)]

            if n == 0 or len(window) == 1:
                indices = [i for t in window for i in time_to_ixs[t]]
                if len(indices) == 0:
                    continue

                X_n = sparse.vstack([X[i] for i in indices])
                S_n = cosine_similarity(X_n)
                n_items = len(indices)
                for i_x, i_n in zip(indices, range(n_items)):
                    for j_x, j_n in zip(indices, range(i_n + 1, n_items)):
                        S[i_x, j_x] = S_n[i_n, j_n]
            else:
                # prev is actually prev + new
                prev_indices = [i for t in window for i in time_to_ixs[t]]
                new_indices = time_to_ixs[window[-1]]

                if len(new_indices) == 0:
                    continue

                X_prev = sparse.vstack([X[i] for i in prev_indices])
                X_new = sparse.vstack([X[i] for i in new_indices])
                S_n = cosine_similarity(X_prev, X_new)
                n_prev, n_new = len(prev_indices), len(new_indices)
                for i_x, i_n in zip(prev_indices, range(n_prev)):
                    for j_x, j_n in zip(new_indices, range(n_new)):
                        S[i_x, j_x] = S_n[i_n, j_n]

        return sparse.csr_matrix(S)


############################### CLUSTER RANKING ################################


class ClusterRanker:
    def rank(self, clusters, collection, vectorizer, plug):
        raise NotImplementedError


class ClusterSizeRanker(ClusterRanker):
    def rank(self, clusters, collection=None, vectorizer=None, plug=False):
        if plug:
            count_dict = {
                i: (
                    len(cluster),
                    plugin.get_page_sum_from_cluster(cluster) / len(cluster),
                )
                for i, cluster in enumerate(clusters)
            }
            ranked = plugin.get_combined_1st_rank(
                count_dict, page_weight=plug, output_one=False
            )
            return [clusters[i] for i, _ in ranked]
        else:
            return sorted(clusters, key=len, reverse=True)


class ClusterDateMentionCountRanker(ClusterRanker):
    def rank(self, clusters, collection=None, vectorizer=None, plug=False):
        date_to_count = collections.defaultdict(int)
        for a in collection.articles():
            for s in a.sentences:
                d = s.get_date()
                if d:
                    date_to_count[d] += 1

        clusters = sorted(clusters, reverse=True, key=len)

        def get_count(c):
            if plug:
                t = c.most_mentioned_time_with_influence_rank(plug)
            else:
                t = c.most_mentioned_time()
            if t:
                return date_to_count[t.date()]
            else:
                return 0

        clusters = sorted(
            clusters, reverse=True, key=get_count
        )  # to give each cluster a specific date
        if plug:
            count_dict = {
                i: (
                    len(cluster),
                    plugin.get_page_sum_from_cluster(cluster) / len(cluster),
                )
                for i, cluster in enumerate(clusters)
            }
            ranked = plugin.get_combined_1st_rank(
                count_dict, page_weight=plug, output_one=False
            )
            return [clusters[i] for i, _ in ranked]
        else:
            return sorted(clusters, key=len, reverse=True)


#
