# coding=utf-8
from __future__ import division
from matplotlib import pyplot as plt
import datetime,random,time
import numpy as np

print "---第二題---"

x = lambda pi,r : pi*r*r

print x(3.14,5)


print "---第三題---"


winning_numbers = random.sample(range(10), 10)
winning_numbers2 = random.sample(range(10), 10)
winning_numbers3 = random.sample(range(10), 10)
print winning_numbers
print winning_numbers2
print winning_numbers3





print "---第四題---"

date = [2015-1-10,2015-1-11,2015-1-12,2015-1-13,2015-1-14,2015-1-15,2015-1-16,2015-1-17,2015-1-18,2015-1-19,2015-1-20]



temp = [16.7,17.4,17.1,20.3,16.2,16.1,17.5,15.3,16.8,16,18.4]

xs = [i for i, _ in enumerate(date)]

plt.xticks(xs,('2015-1-10','2015-1-11','2015-1-12','2015-1-13','2015-1-14','2015-1-15','2015-1-16','2015-1-17','2015-1-18','2015-1-19','2015-1-20'))


plt.plot(xs,temp,color='green', marker='o',linestyle='solid')
plt.title("Taipei January Temperture")
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.show()


print "---第五題---"



A=np.random.randint(1, 10, (2, 3))
B=np.random.randint(1, 10, (2, 3))
C=np.random.randint(1, 10, (2, 3))



def shape(A):
    num_rows = len(A)
    num_cols = len(A[0])
    return num_rows, num_cols

def make_matrix(num_rows, num_cols, entry_fn):
    return [[entry_fn(i, j) for j in range(num_cols)]
            for i in range(num_rows)]

def matrix_add(A, B, C):
    if shape(A) != shape(B) != shape(C):
        raise ArithmeticError("cannot add matrices with different shapes")
    num_rows, num_cols = shape(A)
    def entry_fn(i, j):
        return A[i][j] + B[i][j] + C[i][j]
    return make_matrix(num_rows, num_cols, entry_fn)

def matrix_sub(A, B):
    if shape(A) != shape(B):
        raise ArithmeticError("cannot add matrices with different shapes")
    num_rows, num_cols = shape(A)
    def entry_fn(i, j):
        return A[i][j] - B[i][j]
    return make_matrix(num_rows, num_cols, entry_fn)

print matrix_sub(A,B) #A-B
print matrix_add(A,B,C) #A+B+C




print "---第六題---"
# 填空題
a1=0
a2=0
aboth=0
n=100000
def random_ball():
    return random.choice(["B", "Y"])
random.seed(2)
for _ in range(n):
    get1 = random_ball()
    get2 = random_ball()
    if get1 == "B":
        a1 += 1
    if get1 == "B" and get2 == "B":
        aboth += 1
    if get2 == "B":
        a2 += 1

print "P(aboth):",aboth/n
print "P(get1): ",a1/n
print "P(get2): ",a2/n
print "P(get1,get2): ",a1*a2/n/n
print "P(get1|get2) = p(aboth)/p(get2): ",(aboth/n)/(a2/n)
print "p(get1|get2)/p(get2) = p(get1)p(get2)/p((get2) = p(get1) : ",a1/n



