import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import image
from matplotlib import pyplot
import math
x = []
data1 = np.load('examples.npz', allow_pickle=True)
images1 = data1['images']
df = pd.read_csv('results2.csv',delimiter = ',')
list_of_rows = [list(row) for row in df.values]
list_of_rows.insert(0, df.columns.to_list())
for i in list_of_rows:
	p = []
	for j in i:
		l = j.replace("[  ","[").replace("[ ","[").replace(" ]","]").replace("[","").replace("]","")
		p.append([float(l[:l.index(" ")]),float(l[l.index(" ")+1:])])
	x.append(p)
x = np.array(x)
for u in range(6):
	im = Image.fromarray(images1[u])
	image = Image.open("m1.png")
	image_width, image_height = image.size
	w = math.floor(x[u][54][0]-x[u][48][0])
	h1 = math.floor(x[u][50][1]-x[u][33][1])
	image_resized = image.resize((w, h1), Image.LANCZOS)
	im.paste(image_resized,(math.floor(x[u][48][0]),(math.floor(x[u][33][1]))),image_resized)
	im.save("kk"+f'{u}'+".png")