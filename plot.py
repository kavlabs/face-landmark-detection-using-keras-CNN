import pandas as pd
import matplotlib.pyplot as plt
df  = pd.read_csv("history.csv")
df.plot(x='k',y=['loss','val_loss'])
plt.show()