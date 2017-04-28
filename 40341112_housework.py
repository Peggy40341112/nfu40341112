from bs4 import BeautifulSoup
import requests
import matplotlib.pyplot as plt
import numpy as np


def food_info(td):

    title = td.find("h5").a.text

    return {

        "title":title
    }



base_url = "http://www.gohappy.com.tw/shopping/Browse.do?op=vc&sid=12&cid=292180&cp="

foods = []
price_list = []
NUM_PAGES = 4


for page_num in range(1, NUM_PAGES + 1):
    url = base_url + str(page_num)
    print(url)
    soup = BeautifulSoup(requests.get(url).text, 'html5lib')

    for td in soup('li', onmouseover="this.className='onmouseover';"):
        price = None
        price = td.find("span", "price").text
        foods.append(food_info(td))
        price_list.append(int(price.strip().split(':')[-1].replace(',', '')))



for b in foods:
    print("title: " + b['title'])

print

print("price")
print price_list

print

price_count =[0,0,0,0,0,0,0,0,0,0,0]
price_level = ['200','300','400','500','600','700','800','900','1000','1500','1500~']

for i in price_list:
   if i > 1500 :
      price_count[10] += 1
   elif i > 1000:
      price_count[9]+= 1
   elif i > 900:
      price_count[8] += 1
   elif i > 800:
       price_count[7] += 1
   elif i > 700:
       price_count[6] += 1
   elif i > 600:
      price_count[5] += 1
   elif i > 500:
      price_count[4] += 1
   elif i > 400:
      price_count[3] += 1
   elif i > 300:
      price_count[2] += 1
   elif i > 200:
      price_count[1] += 1
   else :
      price_count[0] += 1

x = np.array([0,1,2,3,4,5,6,7,8,9,10])
plt.xticks(x, price_level)
plt.plot(x,price_count,color='purple', marker='o',linestyle='-.')
plt.ylabel("price of data")
plt.xlabel("price level")
plt.title("Price distribution")
plt.show()









