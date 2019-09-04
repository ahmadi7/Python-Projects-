import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from urllib.request import urlopen
from bs4 import BeautifulSoup

url = "https://www.hubertiming.com/results/2018MLK"
html = (urlopen(url))

soup = BeautifulSoup(html)
tile = soup.title
print(tile)
print(tile.text)

links = soup.find_all('a', href = True)
for link in links:
    print(link['href'])
    print(link.get("href"))

allrows = soup.find_all('tr')
print(allrows[0])
for row in allrows:
    row_list = row.find_all("td")
print(row_list)
for cell in row_list:
    print(cell.text)

data = []
allrows1 = soup.find("tr")
for row in allrows1:
    row_list1 = row.find_all("td")
    dataRow = []
    for cell in row_list1:
        dataRow.append(cell1.text)
     data.append(dataRow)

print(data)


df = pd.DataFrame(data)
print(df)
print(df.head())
print(df.tail())


header_list= []
col_header = soup.find_all('th')
for col in col_header:
    header_list.append(col.text)
print(header_list)

df.colummns = header_list
print(df.head)
df.info()
df.shape()


df2 = df.dropna(how='any')
df2.shape
print(df2)



plt.bar(df2['Gender'], df2['ChipTime_Minute'])
plt.ylabel("Gender")
plt.xlabel("ChipTime_Minutes")
print.title("comparison of average minutes run by male and female")


df2.describe(include=[np.number])





















