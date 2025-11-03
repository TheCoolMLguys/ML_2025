import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def simple_preprocess(train_path, test_path):

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Original shapes - Train: {train_df.shape}, Test: {test_df.shape}")

    train_ids = train_df['ID']
    test_ids = test_df['ID']

    y_train = train_df['grade']

    drop_cols = ['ID', 'grade', 'loan_status']

    X_train = train_df.drop(columns=drop_cols, errors='ignore')
    X_test = test_df.drop(columns=drop_cols, errors='ignore')

    original_features = X_train.columns.tolist()

    categorical_cols = X_train.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        all_data = pd.concat([X_train[col], X_test[col]]).astype(str)
        le = LabelEncoder()
        le.fit(all_data)

        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_df = pd.DataFrame(X_train_scaled, columns=original_features)
    X_test_df = pd.DataFrame(X_test_scaled, columns=original_features)

    target_encoder = LabelEncoder()
    y_train_encoded = target_encoder.fit_transform(y_train)

    print(f"Final shapes - Train: {X_train_df.shape}, Test: {X_test_df.shape}")
    print(f"Target classes: {target_encoder.classes_}")

    save_cleaned_data(X_train_df, y_train_encoded, X_test_df,
                      train_ids, test_ids, target_encoder)

    return X_train_df, y_train_encoded, X_test_df, train_ids, test_ids, target_encoder


def save_cleaned_data(X_train_df, y_train, X_test_df, train_ids, test_ids, target_encoder):

    train_cleaned_df = X_train_df.copy()
    train_cleaned_df['ID'] = train_ids.values
    train_cleaned_df['grade'] = target_encoder.inverse_transform(y_train)
    train_cleaned_df['grade_encoded'] = y_train

    test_cleaned_df = X_test_df.copy()
    test_cleaned_df['ID'] = test_ids.values

    train_cleaned_df.to_csv('Exercise1/data/loan_train_preprocessed.csv', index=False)
    test_cleaned_df.to_csv('Exercise1/data/loan_test_preprocessed.csv', index=False)

    target_mapping = pd.DataFrame({
        'encoded': range(len(target_encoder.classes_)),
        'grade': target_encoder.classes_
    })
    target_mapping.to_csv('Exercise1/data/loan_target_mapping.csv', index=False)


if __name__ == "__main__":
    X_train, y_train, X_test, train_ids, test_ids, target_encoder = simple_preprocess(
        'Exercise1/data/loan-10k.lrn.csv',
        'Exercise1/data/loan-10k.tes.csv'
    )