import matplotlib.pyplot as mp
import pandas as pd
import seaborn as sb
  
# import file with data
data = pd.read_csv("cprobs.csv")
  
# prints data that will be plotted
# columns shown here are selected by corr() since
# they are ideal for the plot
corr = data.corr()
scaled_corr = (corr + 1) / 2
# plotting correlation heatmap
dataplot = sb.heatmap(scaled_corr, cmap="YlGnBu", annot=True)
  
# displaying heatmap
mp.show()