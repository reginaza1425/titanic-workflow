import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)

    # Drop kolom tidak penting
    df.drop(columns=['Cabin', 'Name', 'Ticket'], errors='ignore', inplace=True)

    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Encoding
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

    # Feature & target
    X = df.drop(columns=['Survived'])
    y = df['Survived']

    # Scaling
    scaler = StandardScaler()
    X[['Age', 'Fare']] = scaler.fit_transform(X[['Age', 'Fare']])

    # Split
    return train_test_split(X, y, test_size=0.2, random_state=42)
