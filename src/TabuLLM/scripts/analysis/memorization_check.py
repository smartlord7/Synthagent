import pandas as pd, numpy as np, os, sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
R='results/ablation_20260918_114707/synthetic'
D=['breast_cancer_coimbra','chronic_kidney_disease','hepatitis','thoracic_surgery']
arms=['full','backend_gpt_4_1','backend_claude_sonnet_4_6','raw_prompting','structured_output','cllm','gaussian_copula','bayesian_network','cart','arf','ctgan','tvae','tabddpm']
out=[]
for a in arms:
    tr_rates=[];te_rates=[]
    for d in D:
        real=pd.read_csv(f'data/medical/{d}.csv')
        for k in range(1,11):
            f=f'{R}/{a}/{d}_fold{k}.csv'
            if not os.path.exists(f): continue
            try: syn=pd.read_csv(f)
            except Exception: continue
            if syn.empty: continue
            tr,te=train_test_split(real,test_size=0.2,random_state=42+k)
            cols=[c for c in real.select_dtypes('number').columns if c in syn.columns]
            if not cols: continue
            lo=real[cols].min(); rng=(real[cols].max()-lo).replace(0,1)
            sc=lambda x:((x[cols].apply(pd.to_numeric,errors='coerce')-lo)/rng).fillna(0).values
            rs=np.random.RandomState(k)
            trs=tr.iloc[rs.choice(len(tr),len(te),replace=False)]
            s=sc(syn.head(200)); 
            dtr=pairwise_distances(s,sc(trs)).min(1); dte=pairwise_distances(s,sc(te)).min(1)
            tr_rates.append((dtr<0.05).mean()*100); te_rates.append((dte<0.05).mean()*100)
    if tr_rates: out.append((a,len(tr_rates),np.mean(tr_rates),np.mean(te_rates)))
print(pd.DataFrame(out,columns=['arm','folds','near-copy of train subset %','near-copy of test %']).round(2).to_string(index=False))
