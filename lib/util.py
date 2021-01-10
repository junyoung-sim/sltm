
import csv

from pandas_datareader.data import DataReader

def YahooFinance(id="", start="yyyy-mm-dd", end="yyyy-mm-dd"):
    if end != "yyyy-mm-dd":
        download = DataReader(id.lower(), "yahoo", start, end)
    else:
        download = DataReader(id.lower(), "yahoo", start)
    download.to_csv("./data/" + id.lower() + ".csv")
    with open("./data/" + id.lower() + ".csv", "r") as f:
        lines = f.readlines()
    data, dates = list(download['Adj Close']), [line[:10] for line in lines][1:]
    return {"prices": data, "dates": dates}

def write(data=[], path=""):
    try:
        with open(path, "w+") as File:
            csvWriter = csv.writer(File, delimiter=",")
            csvWriter.writerows(data) 
    except Exception as e:
        print("Failed to write into {}:\n{}" .format(path, e))
        return False

def read(path=""):
    data = []
    try:
        with open(path, "r") as File:
            data = [line.replace("\n", "").split(",") for line in File.readlines()]
            for i in range(len(data)):
                for j in range(len(data[i])):
                    data[i][j] = float(data[i][j])
        return data
    except FileNotFoundError:
        print("No data was read as {} does not exist." .format(path))
