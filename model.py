from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from xgboost                import XGBClassifier


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(class_weight='balanced',max_iter=2000)
    model.fit(X_train,y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(class_weight='balanced',n_estimators=100)
    model.fit(X_train,y_train)
    return model

def train_xgboost(X_train, y_train):
    model = XGBClassifier(scale_pos_weight=2.77,n_estimators=100,learning_rate=0.1)
    model.fit(X_train,y_train)
    return model