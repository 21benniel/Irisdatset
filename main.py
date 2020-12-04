
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
iris = datasets.load_iris()
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier()
knn.fit(iris.data, iris.target)



X = iris.data[:, :2]  
Y = iris.data[:, :1]
iris_panda = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

print(iris_panda)
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title("natha vai seen podatha")

plt.scatter(X[:,0],Y[:,0])

plt.xticks(())
plt.yticks(())

plt.show()
# result=knn.predict([[6.7,3.1,5.6,2.4]])
# print(result)