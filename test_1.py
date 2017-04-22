import tensorflow as tf
import pandas as pd
import numpy as np
label_mapping = pd.Series.from_csv('label_names.csv',header=0).to_dict()
n = len(label_mapping)

video_lvl_record = "train-0.tfrecord"

vid_ids = []
labels = []
labels_for_MLP = []
mean_rgb = []
mean_audio = []

i=0

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
    
    mean_rgb.append(tf_example.features.feature['mean_rgb'].float_list.value)
    mean_audio.append(tf_example.features.feature['mean_audio'].float_list.value)

#     i+=1
#     if i>0:
#         break
    
len(labels_for_MLP)
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

X = mean_audio #[[0., 0.], [1., 1.]]
y = labels_for_MLP #[[0, 1, 1], [1, 1, 0], [1, 0, 0]]
clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
clf.fit(X, y)                         
# clf.predict([[2., 2.], [-1., -2.]])
clf.predict([mean_audio[8]])
a = clf.predict([mean_audio[8]])

b = a[0]>0.15 #index 2885

print(np.nonzero(b))