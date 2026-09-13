#!/usr/bin/env python3
"""
=====================================================================
CODE 3 — مقارنة NDR ببدائل الدمج
يعالج ملاحظة المحكّم: 2-6 (أثبت أن NDR ليس مجرد تطبيع عادي)
=====================================================================
يقارن: NDR / متوسط بسيط / متوسط مرجّح / logistic stacking
كلها على نفس تنبؤات OOF => مقارنة عادلة.

الاستخدام:
    python3 03_ndr_vs_fusion_alternatives.py  final_scores_CorpusA.csv
=====================================================================
"""
import csv, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

CSV=sys.argv[1] if len(sys.argv)>1 else 'final_scores.csv'
THR=0.35
W={'m1':0.40,'m2':0.25,'m3':0.20,'m4':0.15}
rows=list(csv.DictReader(open(CSV)))
def is_mal(r): return r['label'].strip().lower()!='benign'
y=np.array([1 if is_mal(r) else 0 for r in rows])
M=np.array([[float(r['m1_score']),float(r['m2_score']),
             float(r['m3_score']),float(r['m4_score'])] for r in rows])

def scores_to_metrics(score, thr):
    pred=score>=thr
    tp=int(((pred==1)&(y==1)).sum()); fn=int(((pred==0)&(y==1)).sum())
    fp=int(((pred==1)&(y==0)).sum()); tn=int(((pred==0)&(y==0)).sum())
    det=tp/(tp+fn) if tp+fn else 0; fpr=fp/(fp+tn) if fp+tn else 0
    prec=tp/(tp+fp) if tp+fp else 0
    f1=2*prec*det/(prec+det) if prec+det else 0
    return det,fpr,f1

# 1) NDR
def ndr(r_):
    active=[(W[m],r_[i]) for i,m in enumerate(['m1','m2','m3','m4']) if not (m=='m1' and r_[i]==0)]
    ws=sum(w for w,_ in active); return sum(w*s for w,s in active)/ws if ws else 0
ndr_scores=np.array([ndr(r_) for r_ in M])
# 2) متوسط بسيط
avg_scores=M.mean(axis=1)
# 3) متوسط مرجّح ثابت (بلا إعادة توزيع)
wvec=np.array([0.40,0.25,0.20,0.15])
wavg_scores=(M*wvec).sum(axis=1)
# 4) logistic stacking (OOF)
lr=LogisticRegression(max_iter=1000,class_weight='balanced')
stack_scores=cross_val_predict(lr,M,y,cv=5,method='predict_proba')[:,1]

print(f"{'Fusion method':<26}{'Det%':>8}{'FPR%':>8}{'F1%':>9}")
print("-"*52)
for name,sc,thr in [("NDR (proposed)",ndr_scores,THR),
                    ("Simple average",avg_scores,0.5),
                    ("Fixed weighted avg",wavg_scores,0.35),
                    ("Logistic stacking (OOF)",stack_scores,0.5)]:
    det,fpr,f1=scores_to_metrics(sc,thr)
    print(f"{name:<26}{det*100:>7.2f} {fpr*100:>7.2f} {f1*100:>8.4f}")
print("\n>>> لو NDR ≈ أو أفضل من البدائل + بلا تدريب => ردّ قوي على 2-6")
