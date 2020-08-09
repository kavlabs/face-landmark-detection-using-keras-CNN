import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
x = []
data1 = np.load('test_images.npz', allow_pickle=True)
images1 = data1['images']
df = pd.read_csv('results.csv',delimiter = ',')
list_of_rows = [list(row) for row in df.values]
list_of_rows.insert(0, df.columns.to_list())
for i in list_of_rows:
	p = []
	for j in i:
		l = j.replace("[  ","[").replace("[ ","[").replace(" ]","]").replace("[","").replace("]","")
		p.append([float(l[:l.index(" ")]),float(l[l.index(" ")+1:])])
	x.append(p)
x = np.array(x)
for n in range(3):
	fig = plt.figure()
	plt.imshow(images1[n])
	plt.plot(x[n][:, 0], x[n][:, 1], '+r')
	ax = plt.gca()
	ax.invert_yaxis()
	plt.show()
	fig.savefig('my_figure_test'+f'{n}'+'.png')