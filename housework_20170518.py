from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plot
clf = svm.SVC(gamma=0.001, C=100.)
digits = datasets.load_digits()
clf.fit(digits.data[:-1], digits.target[:-1])


my_data = [0,0,0,6,8,2,0,0,
           0,0,0,4,12,4,0,0,
           0,0,1,12,12,2,0,0,
           0,3,12,12,12,1,0,0,
           0,0,1,12,12,1,0,0,
           0,0,1,12,12,2,0,0,
           0,0,1,12,12,2,0,0,
           0,0,0,6,12,4,0,0]

my_data_img = [[0,0,0,6,8,2,0,0],
               [0,0,0,4,12,4,0,0],
               [0,0,1,12,12,2,0,0],
               [0,3,12,12,12,1,0,0],
               [0,0,1,12,12,1,0,0],
               [0,0,1,12,12,2,0,0],
               [0,0,1,12,12,2,0,0],
               [0,0,0,6,12,4,0,0]]


result=clf.predict(my_data)

print "predict: " ,result
print "actual: " ,my_data," my_data ans is 1 "

plot.figure(1, figsize=(3, 3))
plot.imshow(my_data_img, cmap=plot.cm.gray_r, interpolation='nearest')
plot.show()

my_data_2 = [0,0,12,15,12,15,0,0,
             0,0,0,1,3,12,0,0,
             0,0,0,2,1,8,0,0,
             0,0,0,3,1,6,0,0,
             0,0,15,12,12,14,0,0,
             0,0,12,1,1,2,0,0,
             0,0,6,12,12,6,0,0,
             0,0,0,12,14,14,12,0]

my_data_2_img = [[0,0,12,15,12,15,0,0],
                 [0,0,0,1,3,12,0,0],
                 [0,0,0,2,1,8,0,0],
                 [0,0,0,3,1,6,0,0],
                 [0,0,15,12,12,14,0,0],
                 [0,0,12,1,1,2,0,0],
                 [0,0,6,12,12,6,0,0],
                 [0,0,12,14,14,12,0,0]]

result2=clf.predict(my_data_2)

print "predict: " ,result2
print "actual: " ,my_data_2," my_data ans is 2 "

plot.figure(1, figsize=(3, 3))
plot.imshow(my_data_2_img, cmap=plot.cm.gray_r, interpolation='nearest')
plot.show()


