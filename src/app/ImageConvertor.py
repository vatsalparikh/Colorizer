from PIL import Image
import numpy as np
import csv

data = []
with open('output11.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        data.append(row)

w, h = 281, 174
w1, h1 = 641, 361
data = np.array(data).reshape(h,w,3).astype(np.uint8)
print(data.shape)
print(data.dtype)

# data = np.zeros((h, w, 3), dtype=np.uint8)
img = Image.fromarray(data, 'RGB')
img.save('output11.png')
img.show()