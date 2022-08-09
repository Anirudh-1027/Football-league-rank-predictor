import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

path="pl_standings.csv"
 
df = pd.read_csv(path)
print(df)

df['GD'] = df['GF'] - df['GA']
X = df['GD'].values.reshape(-1,1)
y = df['Rk']


from sklearn.linear_model import LinearRegression
linearRegression = LinearRegression()
model=linearRegression.fit(X, y)


filename='data.pkl'
outfile = open(filename,'wb')
pickle.dump(model,outfile)