from collections import Counter
import math, random, csv, json, re
from bs4 import BeautifulSoup
import requests

def get_domain(email_address):

    """split on '@' and return the last piece"""

    return email_address.lower().split("@")[-1]

with open('email.txt', 'r') as f:

    domain_counts = Counter(get_domain(line.strip())

    for line in f

    if "@" in line)

print domain_counts


def process(date, symbol, price):
    print(date, symbol, price)

print("tab delimited stock prices:")

with open('tab_delimited_stock_prices.txt', 'rb') as f:
    reader = csv.reader(f, delimiter='\t')
    # reader = csv.reader(codecs.iterdecode(f, 'utf-8'), delimiter='\t')
    for row in reader:
        date = row[0]
        symbol = row[1]
        closing_price = float(row[2])
        process(date, symbol, closing_price)
print
print("colon delimited stock prices:")

with open('colon_delimited_stock_prices.txt', 'rb') as f:
        reader = csv.DictReader(f, delimiter=':')
        # reader = csv.DictReader(codecs.iterdecode(f, 'utf-8'), delimiter=':')
        for row in reader:
            date = row["date"]
            symbol = row["symbol"]
            closing_price = float(row["closing_price"])
            process(date, symbol, closing_price)
print
print("writing out comma_delimited_score.txt")

today_prices = { 'Chinese' : 90.5, 'English' : 41.68, 'Math' : 64.5 }

with open('tab_delimited_score.txt','w') as f:
        writer = csv.writer(f, delimiter='\t')
        for stock, price in today_prices.items():
            writer.writerow([stock, price])

print("BeautifulSoup")
html = requests.get("http://www.example.com").text
soup = BeautifulSoup(html)
print(soup)
print


