import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load Titanic data
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
df = df.dropna()
X = df[['Pclass', 'Age', 'SibSp', 'Fare']]
y = df['Survived']

# Train the model
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Save the trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
