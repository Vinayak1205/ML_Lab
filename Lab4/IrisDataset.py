import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv("iris (1).csv")

X = df.drop('species', axis=1)
y = df['species']

# Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Build Decision Tree
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train,y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test,y_pred))

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test,y_pred))
