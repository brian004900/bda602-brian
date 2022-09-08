import numpy as np
import pandas
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# load Fisher's Iris dataset into a DataFrame
iris_df = pandas.read_csv("iris.data", header=None)
iris_df.columns = [
    "sepal length",
    "sepal width",
    "petal length",
    "petal width",
    "class",
]

# get simple statistic summary using numpy
iris_np = iris_df.drop("class", axis=1)
iris_np1 = np.array(iris_np)
mean = np.mean(iris_np1, axis=0)
Max = np.max(iris_np1, axis=0)
Min = np.min(iris_np1, axis=0)
print(mean)
print(Max)
print(Min)


# trying five different plots
fig1 = px.scatter(
    iris_df,
    x="sepal length",
    y="petal length",
    color="class",
    hover_name="class",
    log_x=True,
    size_max=60,
)
fig1.show()

fig2 = px.histogram(iris_df, x="sepal length", y="sepal width", color="class")
fig2.show()

fig3 = px.violin(
    iris_df,
    y="petal length",
    x="petal width",
    color="class",
    box=True,
    points="all",
    hover_data=iris_df.columns,
)
fig3.show()

fig4 = px.line(iris_df, x="sepal width", y="petal width", color="class")
fig4.show()

fig5 = px.scatter_3d(
    iris_df,
    x="sepal length",
    y="sepal width",
    z="petal length",
    color="class",
    hover_data=iris_df.columns,
)
fig5.update_layout(scene_zaxis_type="log")
fig5.show()


# scikit-learn(random forest)
x = iris_np
y = iris_df["class"]
scaler = StandardScaler()
x_tr = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_tr, y, test_size=0.3)

random_tree = RandomForestClassifier(n_estimators=10)
random_tree.fit(x_train, y_train)

predict = random_tree.predict(x_test)
score = random_tree.score(x_test, y_test)
print(predict)
print(score)


# scikit-learn(k-means)
knn = KNeighborsClassifier(n_neighbors=7, p=2, metric="minkowski")
knn.fit(x_train, y_train)
print(f"Training data accuracy: {(knn.score(x_train, y_train) * 100)}")
print(f"Testing data accuracy: {(knn.score(x_test, y_test) * 100)}")
print(f"K-mean prediction:{knn.predict(x_test)}")


# pipline
print("------------Model(Random Forest) via Pipeline Predictions------------")

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("randomtree", RandomForestClassifier(n_estimators=10)),
    ]
)
pipeline.fit(X_train, y_train)

prediction = pipeline.predict(X_test)
score = pipeline.score(X_test, y_test)
print(f"Predictions: {prediction}")
print(f"score: {score}")
