20170330 阿璇的筆記
=========================

##ch4線性代數範例

```
ex1 向量運算

//一般情況下直接相加，會變成串接
x = [1,2,3,4,5,6]
y = [4,5,6,5,6]
print x+y

#結果:[1,2,3,4,5,6,4,5,6,5,6]

-----------------------
//如果要進行向量相加，可使用numpy套件
import numpy as np //import np套件
a = np.array([1, 2, 3])
b = np.array([2, 4, 6])
print a+b

#結果:[3 6 9]
```
```
ex2 計算夾角

import numpy as np

a=np.array([1,3,2,2,4])
b=np.array([-2,1,-1,1,9])

la=np.sqrt(a.dot(a))
lb=np.sqrt(b.dot(b))
print("----計算向量長度----")
print (la,lb)

cos_angle=a.dot(b)/(la*lb)

print("----計算cos值----")
print (cos_angle)

angle=np.arccos(cos_angle)

print("----計算夾角(弧度)----")
print (angle)

angle2=angle*360/2/np.pi
print("----弧度轉換成角度----")
print (angle2)

```

```
ex3 比較array與matrix的差別

import numpy as np

#利用array建立矩陣

a=np.array([[3, 4], [2, 3]])
b=np.array([[1, 2], [3, 4]])

#利用matrix建立矩陣

c=np.mat([[1, 3], [5, 1]])
d=np.mat([[6, 1], [3, 5]])

#點乘

e=np.dot(a,b)
f=np.dot(c,d)
print("----乘法運算----")
print (a*b)
print (c*d)
print("----矩陣相乘----")
print (e)
print (f)

#兩者在乘法運算結果不同，而在矩陣相乘時相同

```

```
ex4 利用亂數建立矩陣

import numpy as np

a=np.random.randint(1, 10, (2, 5)) #1:最小值，10:最大值(但不會出現10)，(2,5):空間，2x5的矩陣

print (a)

```

``` 
ex5 行列式

from numpy import *

a = mat([[5,1,-4],[2,6,0],[2,3,6]])
  
print linalg.det(a)

```

```
ex6 配合matplotlib畫出sin的圖形

import numpy as np
from matplotlib import pyplot

x = np.arange(0,11,0.2)
y = np.sin(x)
pyplot.plot(x,y)
pyplot.show()

```

##ch4線性代數額外範例

```
ex7 伴隨矩陣

from numpy import *

A = mat([[-3,2,-5],[-1,0,-2],[3,-4,1]])

print adjugate(A)

```

```
ex8 利用伴隨矩陣算出反矩陣

import numpy as np

A = mat([[-3,2,-5],[-1,0,-2],[3,-4,1]])
B = adjugate(A)/np.linalg.det(A) 

print B
```

資料來源:https://puremonkey2010.blogspot.tw/2012/07/numpy.html

##ch5 統計範例

```
ex1 印出資料

# coding=UTF-8

import pandas as pd
import numpy as np

num_friends = pd.Series([100,49,42,40,54,20,20,19,19,18,18,16,15,15,15,15,14,14,13,13,13,13,12,12,11,10,10,10,10,10,10,10,10,10,10,10,
10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,
2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

num_newFriends =  pd.Series(np.sort(np.random.binomial(203,0.06,204))[::-1])
df_friendsGroup = pd.DataFrame({"A":num_friends,"B":num_newFriends})

print("印出Col A")
print(df_friendsGroup["A"])
print("印出Col A及Col B的前10row")
select = df_friendsGroup[["A", "B"]]
print(select.head(10))
print("印出row5")
print(df_friendsGroup.ix[5])
print("印出row5~row9")
print(df_friendsGroup[5:10])

```

```
ex2 資料讀取與統計量

#資料讀取
dataExcel = pd.read_excel('C:/Users/40341127/Downloads/test.xls')
print(dataExcel)

#統計相關計算

print("最大值= {}".format(df_friendsGroup["A"].max()))
print("最小值= {}".format(df_friendsGroup["A"].min()))
print("平均值= {}".format(df_friendsGroup["A"].mean()))
print("變異數= {}".format(df_friendsGroup["A"].var()))
print("標準差= {}".format(df_friendsGroup["A"].std()))
print("中位數= {}".format(df_friendsGroup["A"].median()))

```

```
ex3 統計量、相關係數、共變異數

print(df_friendsGroup.describe()) #統計量
print(df_friendsGroup.corr()) #相關係數
print("cov = {}".format(num_friends.cov(num_newFriends))) #共變異數

```

```
ex4 圖表輸出

import matplotlib.pyplot as plt

plt.hist(df_friendsGroup["A"],bins=25)
plt.hist(df_friendsGroup["B"],bins=25, color="b")
plt.xlabel("# of Friends")
plt.ylabel("# of People")
plt.show()

```

##ch5統計額外補充

```
ex 利用read_html抓取table資料

import pandas as pd
url = 'http://www.stockq.org/market/asia.php'
table = pd.read_html(url)[4]
print table

```

資料來源:https://jerrynest.io/python-pandas-get-data/