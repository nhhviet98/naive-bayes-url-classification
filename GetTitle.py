import requests
from bs4 import BeautifulSoup
import urllib3
import pandas as pd
import time
import asyncio
import aiohttp


async def get(url):
    try:
        async with aiohttp.ClientSession() as session:
            aiohttp.ClientSession.timeout = 2
            async with session.get(url=url) as response:
                if response.status == 200:
                    resp = await response.read()
                    soup = BeautifulSoup(resp, "html.parser")
                    title = soup.title.string
                    if title == None:
                        title = "None"
                    print(title)
                    return title
                else:
                    print("None")
                    return "None"
    except Exception as e:
        print("None")
        return "None"
        #print("Unable to get url {} due to {}.".format(url, e.__class__))


async def main(urls, amount):
    ret = await asyncio.gather(*[get(url) for url in urls])
    return ret


if __name__ == '__main__':
    df = pd.read_csv("data/URL Classification.csv", names=["idx", "URL", "Category"])
    X = df['URL'].to_list()
    Y = df['Category'].to_numpy()
    #X = X[:1000]
    amount = len(X)
    range_list = [0, 100000, 200000, 300000, 400000, 500000, 600000,
                  700000, 800000, 900000, 1000000, 1100000, 1200000,
                  1300000, 1500000, 1500000]
    for i in range(len(range_list)):
        start = time.time()
        if i != len(range_list) - 1:
            print(f"run from element {range_list[i]} to {range_list[i + 1]}")
            X_temp = X[range_list[i]: range_list[i+1]]
        else:
            print(f"run from element {range_list[i]} to end")
            X_temp = X[len(range_list) - 1:]
        title_list = asyncio.run(main(X_temp, amount))
        file_name = "data/title-" + str(range_list[i]) + ".csv"
        pd.DataFrame(title_list).to_csv(file_name)
        print(f"save to {file_name}")
        end = time.time()
        print(f"Time to run {end - start}")
