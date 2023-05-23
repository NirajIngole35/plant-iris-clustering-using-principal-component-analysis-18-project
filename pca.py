import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt

ds=datasets.load_iris ()
#ds=pd.read_csv(r"C:\Users\HP\Desktop\INTERNSHIP 5 MARCH\ml project\plant iris clustering using principal component analysis 18 project\Iris.csv")
print(ds)


x=ds.data
print(ds)
y=ds.target
name=ds.target_names
print(name)

#fiting pca clustring to the data set with=2
from sklearn.decomposition import PCA
model=PCA(n_components=2)
y_means=model.fit(x).transform(x)
print(y_means)
#variance percentage
plt.figure()
colors=["red",'blue',"green"]

for colors,i  , target_names in zip(colors,[0,1,2],name):
    plt.scatter(y_means[y==i,0],y_means[y==i,1],color=colors,label=target_names)
    plt.show()
