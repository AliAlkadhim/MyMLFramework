import pandas as pd
from sklearn import model_selection

#k fold cross validation

if __name__ == "__main__":
    df = pd.read_csv("input/train.csv")
    #make a dummy column 'k fold' and assign it a value of 01
    df["kfold"] =-1

    #shuffle the data, (and reset the index and drop the index)
    df = df.sample(frac=1).reset_index(drop=True)

kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False)

for fold, (train_indx, val_indx) in enumerate(kf.split(X=df, y=df.target.values)):
    print(len(train_indx), len(val_indx))
    df.loc[val_indx, 'kfold'] = fold
    
df.to_csv("input/train_folds.cvs", index=False)