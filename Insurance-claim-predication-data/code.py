# --------------
# import the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

# Code starts here
df = pd.read_csv(path)
print(df.head())
X = df.drop('insuranceclaim', axis=1)
y = df['insuranceclaim']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=6)
# Code ends here


# --------------


# Code starts here
fig = plt.figure(figsize=(12, 7))
sns.boxplot(X_train["bmi"])

q_value = X_train['bmi'].quantile(0.95)

y_train.value_counts(normalize=True)
# Code ends here


# --------------
# Code starts here
relation = X_train.corr()
print(relation)
import seaborn as sns
sns.pairplot(X_train)

# Code ends here


# --------------

# Code starts here
cols = list(X_train[['children','sex','region','smoker']])

fig, axes = plt.subplots(2,2, figsize=(10,10))

for i in range (0,2):
    for j in range (0,2):
        col = cols[ i * 2 + j]
        axes[i,j].set_title(col)
        sns.countplot(x = X_train[col], hue = y_train, ax=axes[i,j])
        axes[i,j].set_xlabel(col)
        axes[i,j].set_ylabel('insurenceclaim')
plt.show()

# Code ends here

# --------------

# parameters for grid search
parameters = {'C':[0.1,0.5,1,5]}

# Code starts here

lr = LogisticRegression(random_state=9)
grid = GridSearchCV(estimator=lr, param_grid=parameters)

grid.fit(X_train,y_train)
y_pred = grid.predict(X_test)

accuracy = accuracy_score(y_pred,y_test)
print(accuracy)


# Code ends here

# --------------

# Code starts here
score = roc_auc_score(y_test, y_pred)

y_pred_proba = grid.predict_proba(X_test)[:,1]

fpr, tpr, _ = metrics.roc_curve(y_test,y_pred_proba)

roc_auc = roc_auc_score(y_test,y_pred_proba)

plt.plot(fpr,tpr,label='Logistic model,auc='+str(roc_auc))
plt.legend(loc=4)
plt.show()

print(round(score,2))
print(round(y_pred_proba[0],2))
print(round(roc_auc,2)

# Code ends here


