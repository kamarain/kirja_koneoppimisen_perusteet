import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.tree import export_text
import matplotlib.pyplot as plt
plt.style.use('book.mplstyle')

file = "PlayTennis.dat"

X = []
y = []

with open(file, "r", encoding='ISO-8859-1') as f:
    for line in f:
        
        # Skip first line
        if line.startswith("Day"):
            print(line)
            parts = line.strip().split(",")
            feat_names = parts[1:5]
            continue
        
        parts = line.strip().split(",")
        
        outlook = parts[1]
        temperature = parts[2]
        humidity = parts[3]
        wind = parts[4]
        playtennis = parts[5]
        
        X.append([outlook, temperature, humidity, wind])
        y.append(playtennis)
        
X = np.array(X)
y = np.array(y)
print(X)
print(y)

# Encode data to numeric values
encoded_cols = [0, 1, 2, 3]

X_b = np.zeros(X.shape)

for col in encoded_cols:
    lb = LabelEncoder()
    z = lb.fit_transform(X[:, col])
    X_b[:,col] = z
    le_name_mapping = dict(zip(X[:, col], lb.transform(X[:, col])))
    print(feat_names[col])
    print(le_name_mapping)

y_b = lb.fit_transform(y[:])

X_b = X_b.astype(float)
y = y_b.astype(float)
print(X_b)
print(y)

print(feat_names)
# Make a classifier
clf = tree.DecisionTreeClassifier(criterion="entropy",splitter="best")

# Train the classifier using fit function
clf = clf.fit(X_b, y)

# Let's test with training data
print(clf.predict(X_b))

# Print the tree
r = export_text(clf, feature_names=feat_names)
print(r)

# Plot the tree
tree.plot_tree(clf)
plt.show()
