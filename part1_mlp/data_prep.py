import torch
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


def load_and_prepare(batch_size=32, random_state=42):
    raw = load_breast_cancer()
    df = pd.DataFrame(raw.data, columns=raw.feature_names)
    df["target"] = raw.target

    assert df.isnull().sum().sum() == 0, "Unexpected missing values"
    n_dupes = df.duplicated().sum()
    print(f"Duplicate rows: {n_dupes}")
    print(f"Class distribution: {df['target'].value_counts().to_dict()}")

    X = df.drop("target", axis=1).values
    y = df["target"].values

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    to_tensor = lambda x, y: (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    X_tr, y_tr = to_tensor(X_train, y_train)
    X_v, y_v = to_tensor(X_val, y_val)
    X_te, y_te = to_tensor(X_test, y_test)

    print(f"X_train: {X_tr.shape}, y_train: {y_tr.shape}")
    print(f"X_val:   {X_v.shape},  y_val:   {y_v.shape}")
    print(f"X_test:  {X_te.shape}, y_test:  {y_te.shape}")

    train_ds = TensorDataset(X_tr, y_tr)
    val_ds = TensorDataset(X_v, y_v)
    test_ds = TensorDataset(X_te, y_te)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    counts = np.bincount(y_train)
    class_weights = torch.tensor(
        [len(y_train) / (2 * c) for c in counts], dtype=torch.float32
    )
    print(f"Class weights: {class_weights}")

    return train_loader, val_loader, test_loader, class_weights, scaler


if __name__ == "__main__":
    load_and_prepare()
