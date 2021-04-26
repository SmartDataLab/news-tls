#%%
from news_tls.data import Dataset
flag = 0
dataset = Dataset('/data1/su/app/text_forecast/data/datasets_acl/crisis/')
for col in dataset.collections:
    print(col.name) # topic name
    print(col.keywords) # topic keywords
    a = next(col.articles()) # articles collection of this topic
    print("article pub time", a.time)
    if flag > 10:
        break
    
    for s in a.sentences:
        if flag > 10:
            break
        print(s.raw,s.time)
        flag += 1

    
# %%
