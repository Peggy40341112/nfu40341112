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
<<<<<<< Updated upstream
=======

20170406 筆記
=========================

##CH6 機率課堂練習

'''ex1 計算抽出藍球機率

    def random_ball():
        return random.choice(["B", "Y"]) 
      
    if __name__ == "__main__":
    
        a1 = 0
        a2 = 0
        aboth = 0

        random.seed(2)

        n=10000

        for _ in range(n):
            get1 = random_ball()
            get2 = random_ball()
            if get1 == "B":
                a1 += 1
            if get1 == "B" and get2 == "B":
                aboth += 1
            if get2 == "B":
                a2 += 1

        print "P(both):", aboth/n
        print "P(get1): ", a1/n
        print "P(get2): ",a2/n
        print "P(get1,get2): ",a1*a2/n/n
        print "P(get1|get2)=p(both)/p(get2)= ",(aboth/n)/(a2/n)
        print "P(get1|get2=p(get1,get2)/p(get2)=p(get1)p(get2)/p(get2)=p(get1)= ",a1/n

'''

''' ex2 繪出常態分配圖

    from matplotlib import pyplot as plt
    
    def plot_normal_pdfs(plt):

        xs = [x / 10.0 for x in range(-50, 50)]
        plt.plot(xs,[normal_pdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
        plt.plot(xs,[normal_pdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
        plt.plot(xs,[normal_pdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
        plt.plot(xs,[normal_pdf(x,mu=-1)   for x in xs],'-.',label='mu=-1,sigma=1')
        plt.plot(xs,[normal_pdf(x,sigma=3) for x in xs],':',label='mu=0,sigma=3')
        plt.plot(xs,[normal_pdf(x,mu=-2)   for x in xs],'-.',label='mu=-2,sigma=1')
        plt.legend()
        plt.show()  

    plot_normal_pdfs(plt)

'''

##ch6 機率範例練習

''' ex1 sympy-常態分配圖

    # -*- coding: utf-8 -*-
    #import需要的套件
    import numpy as np
    from sympy import *
    import matplotlib.pyplot as plt
    
    fig = plt.gcf()          #獲得當前圖表的instance
    fig.set_size_inches(8,5) #設定圖表的尺寸
    
    # 將 'x' 設為 sympy 可處理的變數
    var('x')
    # 將 x 正規化成lambda函數 f
    f = lambda x: exp(-x**2/2)
    
    # 產生100個介於-4 ~ 5之間,固定間距的連續數
    x = np.linspace(-4,5,100)
    # 依序把 x內的值 丟到 f內 將回傳的值丟回 y
    y = np.array([f(v) for v in x],dtype='float')
    
    plt.grid(True)      #顯示圖表背景格線
    plt.title('sympy常態圖') #圖表標題
    plt.xlabel('X')     #X軸標籤名
    plt.ylabel('Y')     #Y軸標籤名
    plt.plot(x,y,color='blue')  #依 x,y 畫出灰色的線
    plt.fill_between(x,y,0,color='#f0123c') #依 x,y 用綠色填滿到 x軸 之間的空間
    plt.show()

'''

''' ex2 scipy-常態分配圖

    # -*- coding: utf-8 -*-
    #import需要的套件
    from scipy.stats import norm
    import numpy as np
    import matplotlib.pyplot as plt
    
    # subplots()會回傳兩個物件　依序是fig,ax
    # fig為圖表物件
    # ax為圖表內的每張圖
    fig, ax = plt.subplots(1, 1)
    
    # linspace(A,B,size)
    # 會回傳 size 個，從 A 到 B 間距相同的連續數值
    x = np.linspace(norm.ppf(0.02),norm.ppf(0.99), 100)
    
    #這是紅色那條半透明的曲線
    ax.plot(x, norm.pdf(x),'r-', lw=5,alpha=0.6,label='norm pdf')
    
    #這是黑色那條不透明曲線
    ax.plot(x, norm.pdf(x), 'k-', lw=2, label='frozen pdf')
    
    #隨機產生1000個樣本數
    r = norm.rvs(size=10000)
    
    #畫半透明的直方圖
    ax.hist(r, normed=True, histtype='stepfilled', alpha=0.5)
    
    #把有標籤的曲線 統整到圖例中 並列出來
    ax.legend(loc='best', frameon=False)
    
    #把圖表show出來
    plt.show()

'''

##ch7 假設檢定課堂練習

''' ex 假設檢定練習

    from probability_0406 import normal_cdf, inverse_normal_cdf
    import math, random
    
    def normal_approximation_to_binomial(n, p):
        """finds mu and sigma corresponding to a Binomial(n, p)"""
        mu = p * n
        sigma = math.sqrt(p * (1 - p) * n)
        return mu, sigma
    
    def normal_upper_bound(probability, mu=0, sigma=1):
        """returns the z for which P(Z <= z) = probability"""
        return inverse_normal_cdf(probability, mu, sigma)
    
    def normal_lower_bound(probability, mu=0, sigma=1):
        """returns the z for which P(Z >= z) = probability"""
        return inverse_normal_cdf(1 - probability, mu, sigma)
    
    def normal_two_sided_bounds(probability, mu=0, sigma=1):
        """returns the symmetric (about the mean) bounds
        that contain the specified probability"""
        tail_probability = (1 - probability) / 2
    
        # upper bound should have tail_probability above it
        upper_bound = normal_lower_bound(tail_probability, mu, sigma)
    
        # lower bound should have tail_probability below it
        lower_bound = normal_upper_bound(tail_probability, mu, sigma)
    
    
    
    if __name__ == "__main__":
    
        p=0.99
        a=0.46
        mu_0, sigma_0 = normal_approximation_to_binomial(1000, a)
        print("mu_0", mu_0)
        print("sigma_0", sigma_0)
        print("normal_two_sided_bounds("+str(p)+", mu_0, sigma_0)", normal_two_sided_bounds(p, mu_0, sigma_0))
        print
    
        p=0.95
        a=0.46
        mu_0, sigma_0 = normal_approximation_to_binomial(1000, a)
        print("mu_0", mu_0)
        print("sigma_0", sigma_0)
        print("normal_two_sided_bounds("+str(p)+", mu_0, sigma_0)", normal_two_sided_bounds(p, mu_0, sigma_0))
        print
    
        p=0.9
        a=0.46
        mu_0, sigma_0 = normal_approximation_to_binomial(1000, a)
        print("mu_0", mu_0)
        print("sigma_0", sigma_0)
        print("normal_two_sided_bounds("+str(p)+", mu_0, sigma_0)", normal_two_sided_bounds(p, mu_0, sigma_0))
        print

'''

##ch7 假設檢定範例練習

'''

    from __future__ import division
    import math, random
    
    
    if __name__ == "__main__":
        # coding=utf-8
        #from __future__ import division
        #import math, random
        #第一題 假設檢定
        #設定mu = 98 , sigma = 10
        mu_0, sigma_0 = 98,10
        print "mu_0", mu_0
        print "sigma_0", sigma_0
    
    
        # 常態函數分布累積method
        def normal_cdf(x, mu=0, sigma=1):
            return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2
    
        #逼近Z值method
        def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.00001):
            """find approximate inverse using binary search"""
            # if not standard, compute standard and rescale
            if mu != 0 or sigma != 1:
                return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)
    
            low_z, low_p = -10.0, 0  # normal_cdf(-10) is (very close to) 0
            hi_z, hi_p = 10.0, 1  # normal_cdf(10)  is (very close to) 1
            while hi_z - low_z > tolerance:
                mid_z = (low_z + hi_z) / 2  # consider the midpoint
                mid_p = normal_cdf(mid_z)  # and the cdf's value there
                if mid_p < p:
                    # midpoint is still too low, search above it
                    low_z, low_p = mid_z, mid_p
                elif mid_p > p:
                    # midpoint is still too high, search below it
                    hi_z, hi_p = mid_z, mid_p
                else:
                    break
    
            return mid_z
    
        #求Z值的method
        def normal_lower_bound(probability, mu=0, sigma=1):
            """returns the z for which P(Z >= z) = probability"""
            return inverse_normal_cdf(1 - probability, mu, sigma)
    
        #第一個問題
        # a = 2.5% = 0.75機率
        print "normal_lower_bound(0.75, mu_0, sigma_0)", normal_lower_bound(0.75, mu_0, sigma_0)
        print
    
        #第一題第二個問題
        print "normal_lower_bound(0.9, mu_0, sigma_0) = ", normal_lower_bound(0.9, mu_0, sigma_0)
    
        #第一題第三個問題，求顯著性
        #normal_probability_below = normal_cdf
        print "normal_probability_belowx,mu,sigma) = ", normal_cdf(81.55,98, 10)
        print ("")
    
        #第二題 信賴區間
    
        #右尾檢定
        def normal_upper_bound(probability, mu=0, sigma=1):
            """returns the z for which P(Z <= z) = probability"""
            return inverse_normal_cdf(probability, mu, sigma)
    
        #左尾檢定
        def normal_lower_bound(probability, mu=0, sigma=1):
            """returns the z for which P(Z >= z) = probability"""
            return inverse_normal_cdf(1 - probability, mu, sigma)
    
        #雙尾檢定
        def normal_two_sided_bounds(probability, mu=0, sigma=1):
            """returns the symmetric (about the mean) bounds
            that contain the specified probability"""
            tail_probability = (1 - probability) / 2
    
            # upper bound should have tail_probability above it
            upper_bound = normal_lower_bound(tail_probability, mu, sigma)
    
            # lower bound should have tail_probability below it
            lower_bound = normal_upper_bound(tail_probability, mu, sigma)
    
            return lower_bound, upper_bound
    
        print "Confidence Intervals"
        print "normal_two_sided_bounds(信賴水準,平均數,標準差) = ",normal_two_sided_bounds(0.95,4.015,0.02)
    
        #第三題 A/B Testing
    
        #計算p(期望值/平均數) sigma(標準差)
        def estimated_parameters(N, n):
            p = n / N
            sigma = math.sqrt(p * (1 - p) / N)
            return p, sigma
    
        #計算兩者差距
        def a_b_test_statistic(N_A, n_A, N_B, n_B):
            p_A, sigma_A = estimated_parameters(N_A, n_A)
            p_B, sigma_B = estimated_parameters(N_B, n_B)
            return (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B ** 2)
    
        z = a_b_test_statistic(1500, 400, 1400, 350)
        print "PB-PA間的差距",(z)
    
        #計算p-value值
    
        normal_probability_below = normal_cdf
    
    
        # it's above the threshold if it's not below the threshold
        def normal_probability_above(lo, mu=0, sigma=1):
            return 1 - normal_cdf(lo, mu, sigma)
        
        def two_sided_p_value(x, mu=0, sigma=1):
            if x >= mu:
                # if x is greater than the mean, the tail is above x
                return 2 * normal_probability_above(x, mu, sigma)
            else:
                # if x is less than the mean, the tail is below x
                return 2 * normal_probability_below(x, mu, sigma)
    
        print "檢定兩者之間是否有差異",(two_sided_p_value(z))

'''

