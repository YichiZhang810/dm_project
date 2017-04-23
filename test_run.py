import tensorflow as tf
import pandas as pd
import numpy as np

################  Get and store all info from train-0.tfrecord ##################
label_mapping = pd.Series.from_csv('label_names.csv',header=0).to_dict()
n = len(label_mapping)

video_lvl_record = "train-0.tfrecord"

vid_ids = []
labels = []
labels_for_MLP = []
mean_rgb = []
mean_audio = []

textual_labels = []
textual_labels_nested = []

# i=0

for example in tf.python_io.tf_record_iterator(video_lvl_record):
    tf_example = tf.train.Example.FromString(example) # get visualized TFRecord
    vid_ids.append(tf_example.features.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8'))
    
    array = np.zeros(n)
    tmp_labels=tf_example.features.feature['labels'].int64_list.value
    tmp_labels_after_pp = []
    for x in tmp_labels:
        if x<4716:
            tmp_labels_after_pp.append(x)
    labels.append(tmp_labels_after_pp)
    array[tmp_labels]=1
    labels_for_MLP.append(array)

    label_example_textual = [label_mapping[x] for x in tmp_labels]
    textual_labels_nested.append(set(label_example_textual))
    textual_labels = textual_labels + label_example_textual
    
    mean_rgb.append(tf_example.features.feature['mean_rgb'].float_list.value)
    mean_audio.append(tf_example.features.feature['mean_audio'].float_list.value)


#################### Get distance(1-correlation) matrix for all labels #######################
def grouped_data_for(l):
    # wrap the grouped data into dataframe, since the inner is pd.Series, not what we need
    l_with_c = pd.DataFrame(
        pd.DataFrame({'label': l}).groupby('label').size().rename('n')
    ).sort_values('n', ascending=False).reset_index()
    return l_with_c

textual_labels_with_counts_all = grouped_data_for(textual_labels)
top_50_labels = list(textual_labels_with_counts_all['label'][0:50].values)

# get all unique labels
all_unique_labels = list(textual_labels_with_counts_all['label'].values)

K_labels = []

for i in all_unique_labels: #top_50_labels:
    row = []
    for j in all_unique_labels: #top_50_labels:
        # find all records that have label `i` in them
        i_occurs = [x for x in textual_labels_nested if i in x]
        # how often does j occur in total in them?
        j_and_i_occurs = [x for x in i_occurs if j in x]
        k = 1.0*len(j_and_i_occurs)/len(i_occurs)
        row.append(k)
    K_labels.append(row)

K_labels = np.array(K_labels)
K_labels = pd.DataFrame(K_labels)
K_labels.columns = all_unique_labels
K_labels.index = all_unique_labels
K_labels_trans = K_labels.transpose()
K_labels_new = (K_labels + K_labels_trans)/2
K_labels_dist = 1-K_labels_new

######################## Use linkage() and dendrogram by generating condense matrix ####################
import scipy.spatial.distance as ssd
# convert the redundant n*n square matrix form into a condensed nC2 array
distArray = ssd.squareform(K_labels_dist) # condense dist matrix

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
Z = linkage(distArray)

def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    ddata = dendrogram(*args, **kwargs)

    if max_d:
        plt.axhline(y=max_d, c='k')
    return ddata


# plt.figure(figsize=(25, 10))
# max_d = 0.4
# fancy_dendrogram(
#     Z,
# #     truncate_mode='lastp',
# #     p=12,
#     leaf_rotation=90.,
#     leaf_font_size=12.,
#     max_d =max_d
# #     show_contracted=True,
# #     annotate_above=10,  # useful in small plots so annotations don't overlap
# )
# plt.show()

#################### Clustering #########################
from scipy.cluster.hierarchy import fcluster
# max_d=0.7
k = 650
# clusters = fcluster(Z, max_d, criterion='distance')
clusters = fcluster(Z, k, criterion='maxclust')
# clusters
numOfClusters = len(set(clusters))

import sys
import numpy as np

roots = np.zeros(numOfClusters)

for i in range(1, numOfClusters+1):
    tmp_cluster = np.where(clusters==i)
#     print(tmp_cluster)
    for j in tmp_cluster[0]:
        aggre_dist = 0
        min_dist = sys.float_info.max
        for k in tmp_cluster[0]:
            aggre_dist = aggre_dist + K_labels_dist.iloc[j][k]
        if(aggre_dist < min_dist):
            min_dist = aggre_dist
            roots[i-1] = j

#################### Basic model #######################
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

X = mean_audio #[[0., 0.], [1., 1.]]
y = labels_for_MLP #[[0, 1, 1], [1, 1, 0], [1, 0, 0]]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
clf.fit(X, y)                         
# clf.predict([[2., 2.], [-1., -2.]])
# clf.predict([mean_audio[8]])

a = clf.predict([mean_audio[8]])

################### Trying to modify the prediction by adding related labels ####################
b = np.nonzero(a)
# print(b[1])

res = set()
for label in b[1]:
    res.add(label)
    label_name = label_mapping[label]
    unique_idx = all_unique_labels.index(label_name)
    tmp_cluster = clusters[unique_idx]
    root_label = (int)(roots[tmp_cluster-1])
    addition_label_name = all_unique_labels[root_label]
    res.add(list(label_mapping.values()).index(addition_label_name))

for i in res:
    print(label_mapping[i])


#################### Result so far: #####################
print('--Original labels:')

for i in labels[8]:
    print(label_mapping[i])

print()

print('--Basic model predicting labels:')

for i in b[1]:
    print(label_mapping[i])

print()

print('--Modified model predicting labels:')

for i in res:
    print(label_mapping[i])
