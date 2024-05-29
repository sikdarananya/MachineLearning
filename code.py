import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

music_data =pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])

Y = music_data['genre']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2) #allocating 20% of the data for testing


model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)


score = accuracy_score(Y_test, predictions)
score

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
#import joblib

music_data =pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])

Y = music_data['genre']

model = DecisionTreeClassifier()
model.fit(X, Y)

# model = joblib.load('music-recommended.joblib')
# predictions = model.predict([[21,1]])
# predictions

tree.export_graphviz(model, out_file='music-recomeded.dot',
                    feature_names = ['age', 'gender'],
                    class_names = sorted(Y.unique()),
                    label = 'all',
                    rounded = True,
                    filled = True)

