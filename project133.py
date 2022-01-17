from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
df = pd.read_csv('star_with_gravity.csv')

radius = df['Radius'].to_list()
mass = df['Mass'].to_list()

X=[]
for index, planet_mass in enumerate(mass):
  temp_list = [radius[index], planet_mass]
  X.append(temp_list)
  
wcss = []
for i in range(1, 11):
  kmeans = KMeans(n_clusters = i, init="k-means++", random_state=42)
  kmeans.fit(X)
  wcss.append(kmeans.inertia_)

plt.figure(figsize = (10, 5))
sns.lineplot(range(1, 11), wcss, marker="o", color="Red")
plt.title("Elbow Method")
plt.xlabel("No. of Cluster")
plt.ylabel("wcss")
plt.show() 
