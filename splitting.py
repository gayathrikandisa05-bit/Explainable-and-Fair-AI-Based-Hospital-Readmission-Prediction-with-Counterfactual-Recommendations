"""Patient-disjoint stratified data splits shared by training and evaluation."""
from sklearn.model_selection import StratifiedGroupKFold

def patient_disjoint_split(X, y, groups, seed=42):
    """Return roughly 71/14/14 train/validation/test row indices, keeping patients intact."""
    cv=StratifiedGroupKFold(n_splits=7,shuffle=True,random_state=seed)
    folds=list(cv.split(X,y,groups))
    test_idx=folds[0][1]; valid_idx=folds[1][1]
    excluded=set(test_idx)|set(valid_idx)
    train_idx=[i for i in range(len(X)) if i not in excluded]
    return train_idx,valid_idx,test_idx
