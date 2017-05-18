from sklearn import datasets
from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)
iris = datasets.load_iris()
clf.fit(iris.data[:-10], iris.target[:-10])
result=clf.predict(iris.data[-10:])

print ("predict: \n")
print result
print ("actual: \n")
print (iris.data[-10:])
print (iris.target[-10:])