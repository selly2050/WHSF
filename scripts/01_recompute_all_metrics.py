#!/usr/bin/env python3
"""
=====================================================================
CODE 1 — إعادة حساب كل المقاييس من ملف تنبؤات واحد
يعالج ملاحظات المحكّم: 1-4، 1-5، 2-3
=====================================================================
الاستخدام:
    python3 01_recompute_all_metrics.py  path/to/final_scores_CorpusA.csv

المطلوب في ملف CSV: أعمدة label, m1_score, m2_score, m3_score, m4_score
(label = 'benign' للحميدة، أي شيء آخر = خبيثة)
=====================================================================
"""
import csv, sys

CSV = sys.argv[1] if len(sys.argv)>1 else 'final_scores.csv'
THRESHOLD = 0.35
W = {'m1':0.40, 'm2':0.25, 'm3':0.20, 'm4':0.15}

rows = list(csv.DictReader(open(CSV)))
def is_mal(r): return r['label'].strip().lower() != 'benign'

def ndr_fts(r, use):
    active=[]
    for m in use:
        s=float(r[f'{m}_score'])
        if not (m=='m1' and s==0):      # M1=0 على hash-novel => غير نشطة
            active.append((W[m], s))
    if not active: return 0.0
    wsum=sum(w for w,_ in active)
    return sum(w*s for w,s in active)/wsum

def evaluate(use, thr=THRESHOLD):
    TP=FN=FP=TN=0; missed=[]
    for r in rows:
        pred = ndr_fts(r,use) >= thr
        if is_mal(r):
            if pred: TP+=1
            else: FN+=1; missed.append(r)
        else:
            if pred: FP+=1
            else: TN+=1
    det=TP/(TP+FN) if TP+FN else 0
    fpr=FP/(FP+TN) if FP+TN else 0
    prec=TP/(TP+FP) if TP+FP else 0
    f1=2*prec*det/(prec+det) if prec+det else 0
    return dict(TP=TP,FN=FN,FP=FP,TN=TN,det=det,fpr=fpr,prec=prec,f1=f1,missed=missed)

configs = [
    ("M1 only",        ['m1']),
    ("M1+M2",          ['m1','m2']),
    ("M1+M2+M3",       ['m1','m2','m3']),
    ("WHSF (all 4)",   ['m1','m2','m3','m4']),
]
print(f"{'Configuration':<18}{'Det%':>9}{'FPR%':>9}{'Prec%':>11}{'F1%':>11}  (TP,FN,FP,TN)")
print("-"*72)
res={}
for name,use in configs:
    r=evaluate(use); res[name]=r
    print(f"{name:<18}{r['det']*100:>8.2f} {r['fpr']*100:>8.2f} {r['prec']*100:>10.4f} {r['f1']*100:>10.4f}  ({r['TP']},{r['FN']},{r['FP']},{r['TN']})")

# ملاحظة 1-4/2-3: العيّنة التي يغيّرها M4
before=set(id(x) for x in res['M1+M2+M3']['missed'])
after =set(id(x) for x in res['WHSF (all 4)']['missed'])
print("\n[1-4] أثر إضافة M4:")
d_before=res['M1+M2+M3']; d_after=res['WHSF (all 4)']
print(f"   M1+M2+M3 detection = {d_before['det']*100:.2f}%  ({d_before['FN']} فائتة)")
print(f"   WHSF     detection = {d_after['det']*100:.2f}%  ({d_after['FN']} فائتة)")
delta=d_after['FN']-d_before['FN']
if delta>0:
    print(f"   => M4 زاد الفائتات بمقدار {delta} (على هذا الـcorpus، M3 يفصل وحده)")
elif delta<0:
    print(f"   => M4 قلّل الفائتات بمقدار {-delta} (M4 مفيد على هذا الـcorpus)")
else:
    print(f"   => M4 لم يغيّر عدد الفائتات")
