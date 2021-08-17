import numpy as np
import  matplotlib.pyplot as plt
import pandas as pd
from sklearn import cluster, tree, decomposition
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib as mpl
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics as sm



def label_encoded(feat):
    le = LabelEncoder()
    le.fit(feat)
    #print(feat.name,le.classes_)
#     print(le.classes_)
    return le.transform(feat)

def plotData(df, groupby):

    fig, ax = plt.subplots(figsize = (10,6),dpi= 100)
    ax.set(facecolor = "#aadbf0")

    # color map
    cmap = mpl.cm.get_cmap('prism')

   
    for i, cluster in df.groupby(groupby):
        cluster.plot(ax = ax, # need to pass this so all scatterplots are on same graph
                     kind = 'scatter', 
                     x = 'X', y = 'Y',
                     color = cmap(i/(nclusters-1)), # cmap maps a number to a color
                     label = "%s %i" % (groupby, i), 
                     s=30,edgecolors= "black", linewidth=0.4) # dot size
    ax.grid(color='white')
    ax.axhline(0, color='white')
    ax.axvline(0, color='white')
    ax.legend(loc='upper left',fontsize="small",bbox_to_anchor=(1.05, 1),title="odor")
    ax.set_title("Principal Components Analysis (PCA) of mushroom Dataset");






#main program
filename = 'mushrooms_data.txt'

h_names=['class', 'cap-shape', 'Cap-surface','cap-color', 'bruises', 'odor','gill-attachment', 'gill-spacing', 'gill-size','gill-color', 'stalk-shape', 'stalk-surface-above-ring','stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']
mushrooms = pd.read_csv(filename,names = h_names)
mushrooms = mushrooms.drop('veil-type', axis=1) # get rid of the veil-type column - don't need it

X = mushrooms.drop(['odor'], axis=1)
Y = mushrooms['odor']


for col in X.columns:
    X[str(col)] = label_encoded(X[str(col)])


from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

scaler.fit(X)
X_scaled_array = scaler.transform(X)
X_scaled = pd.DataFrame(X_scaled_array, columns = X.columns)

X_scaled.sample(5)


nclusters = 9 # this is the k in kmeans

km = KMeans(n_clusters=nclusters, random_state=0)
v = km.fit(X_scaled)
# predict the cluster for each data point
y_cluster_kmeans = km.predict(X_scaled)
y_cluster_kmeans


from sklearn import metrics
score = metrics.silhouette_score(X_scaled, y_cluster_kmeans)
print(score)


scores = metrics.silhouette_samples(X_scaled, y_cluster_kmeans)
sns.distplot(scores);

df_scores = pd.DataFrame()
df_scores['SilhouetteScore'] = scores
df_scores['odor'] = mushrooms['odor']
df_scores.hist(by='odor', column='SilhouetteScore', range=(0,1.0), bins=20);
#sns.pairplot(mushrooms, hue="odor", diag_kind="hist", height=1.6);
sns.pairplot(df_scores, hue="odor", size=4);


ndimensions = 2


pca = PCA(n_components=ndimensions, random_state=0)
pca.fit(X_scaled)
X_pca_array = pca.transform(X_scaled)
X_pca = pd.DataFrame(X_pca_array, columns=['X','Y']) # PC=principal component




y_id_array = pd.Categorical(mushrooms['odor']).codes
df_plot = X_pca.copy()
df_plot['ClusterKmeans'] = y_cluster_kmeans
df_plot['odor number'] = y_id_array # actual labels

from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=nclusters)
gmm.fit(X_scaled)

# predict the cluster for each data point
y_cluster_gmm = gmm.predict(X_scaled)
y_cluster_gmm
   
df_plot['ClusterGMM'] = y_cluster_gmm
plotData(df_plot, 'ClusterKmeans')
plotData(df_plot, 'odor number')


plotData(df_plot, 'ClusterGMM')

from sklearn.metrics.cluster import adjusted_rand_score

score1 = metrics.silhouette_score(X_scaled, y_cluster_kmeans)
print(score1)
# first let's see how the k-means clustering did - 
score2 = adjusted_rand_score(Y, y_cluster_kmeans)
print(score2)


#unstable results for GMM Algorithm - K-means wins here even that it's didn't cluster well.
score3 = adjusted_rand_score(Y, y_cluster_gmm)
print(score3)
score4 = metrics.silhouette_score(X_scaled, y_cluster_gmm)
print(score4)


