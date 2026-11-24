"""Phase 1: inspect SPSS files (no modeling). Usage: python -I src/inspect_data.py"""
import pyreadstat, pandas as pd, sys
pd.set_option("display.width",200); pd.set_option("display.max_rows",500)
D="data/"
for f in ["RV20_Mood_MC_20230519.sav","RV20_Questionnaires_MC_20230519.sav","RV20_EmotionRecognition_MC_20230519.sav"]:
    df,meta=pyreadstat.read_sav(D+f)
    print("\n=====",f,df.shape)
    lab=meta.column_names_to_labels
    miss=df.isna().sum()
    rows=[(c,str(df[c].dtype),int(miss[c]),df[c].nunique(),lab.get(c) or "") for c in df.columns]
    print(pd.DataFrame(rows,columns=["col","dtype","na","nuniq","label"]).to_string(index=False))
    print("value labels:",{k:v for k,v in meta.variable_value_labels.items()})
