import pyxdf
import numpy as np
import csv

XDF_FILENAME = 'test3.xdf'
CSV_FILENAME = 'test3.csv'

data, header = pyxdf.load_xdf(XDF_FILENAME)

with open(CSV_FILENAME, 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["Marker", "Timestamp"]
    writer.writerow(field)

    for stream in data:
        y = stream['time_series']

        if isinstance(y, list):
            for timestamp, marker in zip(stream['time_stamps'], y):
                line = f'Marker "{marker[0]}" @ {timestamp:.2f}s'
                row = line.split('" @ ')
                print(line)
                row[0] = row[0][-1]
                row[1] = row[1][:-1]
                if(row[0] != '"'):
                    writer.writerow(row)
