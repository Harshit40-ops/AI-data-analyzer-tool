from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def train_models(df,target):

    X = df.drop(columns=[target])
    y = df[target]

    X = pd.get_dummies(X)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    results = {}

    if y.dtype == "object":

        model1 = LogisticRegression(max_iter=1000)
        model2 = RandomForestClassifier()

        model1.fit(X_train,y_train)
        model2.fit(X_train,y_train)

        p1 = model1.predict(X_test)
        p2 = model2.predict(X_test)

        results["LogisticRegression"] = accuracy_score(y_test,p1)
        results["RandomForest"] = accuracy_score(y_test,p2)

    else:

        model1 = LinearRegression()
        model2 = RandomForestRegressor()

        model1.fit(X_train,y_train)
        model2.fit(X_train,y_train)

        p1 = model1.predict(X_test)
        p2 = model2.predict(X_test)

        results["LinearRegression"] = r2_score(y_test,p1)
        results["RandomForest"] = r2_score(y_test,p2)

    best_model = max(results,key=results.get)

    return best_model,results