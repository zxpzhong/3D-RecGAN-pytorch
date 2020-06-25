import numpy as np
all_feature = np.load('all_feature.npy')
feature_len1  ,feature_len2 , feature_len3  ,feature_len4 = 4728, 4810, 2520, 104
color = np.zeros([feature_len1 + feature_len2 + feature_len3 + feature_len4])
color[:feature_len1] = 1
color[feature_len1:feature_len1 + feature_len2] = 2
color[feature_len1 + feature_len2:feature_len1 + feature_len2 + feature_len3] = 3
color[feature_len1 + feature_len2 + feature_len3:] = 4

# 3. 提取测试集中所有的测试样本的feature
'''t-SNE'''
from time import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets

fig = plt.figure(figsize=(8, 8))
t0 = time()
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
Y = tsne.fit_transform(all_feature)  # 转换后的输出
t1 = time()
print("t-SNE: %.2g sec" % (t1 - t0))  # 算法用时
ax = fig.add_subplot(1, 1, 1)
print(Y.shape)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("t-SNE (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
ax.yaxis.set_major_formatter(NullFormatter())
# plt.axis('tight')
# 保存图像
plt.savefig('./all.png')
# 最终显示
plt.show()