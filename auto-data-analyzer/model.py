from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def train_models(df, target):

    # Separate features and target
    X = df.drop(columns=[target])
    y = df[target]

    # Convert categorical feature columns into numeric
    X = pd.get_dummies(X, drop_first=True)

    # Handle missing values in target
    if y.isnull().sum() > 0:
        if y.dtype == "object":
            y = y.fillna(y.mode()[0])
        else:
            y = y.fillna(y.mean())

    results = {}

    # Classification task
    if y.dtype == "object" or y.nunique() < 10:

        # Encode target labels
        le = LabelEncoder()
        y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model1 = LogisticRegression(max_iter=1000)
        model2 = RandomForestClassifier(random_state=42)

        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)

        p1 = model1.predict(X_test)
        p2 = model2.predict(X_test)

        results["LogisticRegression"] = accuracy_score(y_test, p1)
        results["RandomForestClassifier"] = accuracy_score(y_test, p2)

    # Regression task
    else:

        # Convert target safely to numeric
        y = pd.to_numeric(y, errors="coerce")
        y = y.fillna(y.mean())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model1 = LinearRegression()
        model2 = RandomForestRegressor(random_state=42)

        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)

        p1 = model1.predict(X_test)
        p2 = model2.predict(X_test)

        results["LinearRegression"] = r2_score(y_test, p1)
        results["RandomForestRegressor"] = r2_score(y_test, p2)

    best_model = max(results, key=results.get)

    return best_model, results
