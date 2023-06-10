#載入函數式
from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#載入資料
iris = datasets.load_iris()     
features = iris.data
target = iris.target

############ 資料前置處理(圖表化的資料) ##############
"""
x = pd.DataFrame(iris['data'], columns=iris['feature_names'])
y = pd.DataFrame(iris['target'], columns=['target_names'])
data = pd.concat([x,y], axis=1)
print(data)
"""

############ K最近鄰預測 ##############
standardizer = StandardScaler()   # 產生標準化器
features_standardized = standardizer.fit_transform(features)        # 特徴標準化
nearest_neighbors = NearestNeighbors(n_neighbors=2).fit(features_standardized)  # 2最近鄰
new_observation = [ 1,  1,  1,  1]  # 產生觀察
distances, indices = nearest_neighbors.kneighbors([new_observation])    # 找出距離與觀察之最近鄰點的索引
nearestneighbors_euclidean = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(features_standardized) # 依據歐幾里德距離找出二最近鄰
nearestneighbors_euclidean = NearestNeighbors(n_neighbors=3, metric="euclidean").fit(features_standardized) # 依據歐幾里德距離找出每一觀察的三個最近鄰(包含自己)
nearest_neighbors_with_self = nearestneighbors_euclidean.kneighbors_graph(features_standardized).toarray()  # 列出每個觀察的三個最近鄰(包含自己)
for i, z in enumerate(nearest_neighbors_with_self): z[i] = 0    # 移除讓自己成為最近觀察值的1

# 繪製散佈圖
scatter_sepal = plt.scatter(features_standardized[:, 0], features_standardized[:, 1], c=target, cmap='viridis')
scatter_petal = plt.scatter(features_standardized[:, 2], features_standardized[:, 3], c=target, cmap='viridis')

#關閉原本的圖片
plt.close()

#將圖片切割成兩部分
fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].set_title("KNN predict with scatter kernel")
axs[0].scatter(features_standardized[:, 0], features_standardized[:, 1], c=target, cmap='viridis')
axs[0].set_xlabel("Sepal length")
axs[0].set_ylabel("Sepal width")

axs[1].set_title("KNN predict with scatter kernel")
axs[1].scatter(features_standardized[:, 2], features_standardized[:, 3], c=target, cmap='viridis')
axs[1].set_xlabel("Petal length")
axs[1].set_ylabel("Petal width")

#把圖例跟相對亞屬標籤結合後放在legend
handles, labels = scatter_sepal.legend_elements()
axs[0].legend(handles, iris.target_names, loc='best')

handles, labels = scatter_petal.legend_elements()
axs[1].legend(handles, iris.target_names, loc='best')

fig.tight_layout()
plt.show()

############ SVC進行分類 ##############
#X是花萼與花瓣的長度、寬度的原始資料
#Y是將花分類之後的正確答案
X = iris.data
y = iris.target


X = iris.data[:,:2]
y = iris.target
plt.scatter(X[y==0,0],X[y==0,1],color = 'r',marker='o')
plt.scatter(X[y==1,0],X[y==1,1],color = 'b',marker='o')
plt.scatter(X[y==2,0],X[y==2,1],color = 'g',marker='o')
plt.title('the relationship between sepal and target classes')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()

X = iris.data[:,2:]
y = iris.target
plt.scatter(X[y==0,0],X[y==0,1],color = 'r',marker='o')
plt.scatter(X[y==1,0],X[y==1,1],color = 'b',marker='o')
plt.scatter(X[y==2,0],X[y==2,1],color = 'g',marker='o')
plt.title('the relationship between Petal and target classes')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.show()

#test_size和random_state都是暫定的，可以改
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
lin_svc = svm.SVC(kernel='linear').fit(X_train, y_train)
rbf_svc = svm.SVC(kernel='rbf').fit(X_train, y_train)
poly_svc = svm.SVC(kernel='poly', degree=3).fit(X_train, y_train)

#print('train shape:', x_train.shape)
#print('test shape:', x_test.shape)												   



##### 繪製決策邊界 ######
# 建立網格座標系統(sepal花萼）
"""
x軸、y軸、間隔h（決定網格密集程度）
"""
h = .02 
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


titles = ['SVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']


"""
clf: 已經訓練好的分類器(利用這個對網格座標數據進行預測)
xx: 網格的 X 軸座標數據
yy: 網格的 Y 軸座標數據
params: 一個字典，包含傳遞給 contourf 函數的參數（繪圖參數：色彩、透明度）
"""
for i, clf in enumerate((lin_svc, rbf_svc, poly_svc)):
    plt.subplot(2, 2, i + 1)  #建立一個2*2的網格放置三個圖
    plt.subplots_adjust(wspace=0.5, hspace=0.5)  #圖和圖之間的margin
    #SVM input :xx and yy
    #output: an array
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) 
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8) 
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()


lin_svc_pre = lin_svc.predict(X_test)
acc_lin_svc = sum(lin_svc_pre==y_test)/len(y_test)
rbf_svc_pre = rbf_svc.predict(X_test)
acc_rbf_svc = sum(rbf_svc_pre==y_test)/len(y_test)
poly_svc_pre = poly_svc.predict(X_test)
acc_poly_svc = sum(poly_svc_pre==y_test)/len(y_test)
print(acc_lin_svc)
print(acc_rbf_svc)
print(acc_poly_svc)



# 
X_train, X_test, y_train, y_test = train_test_split(iris.data[:,2:], iris.target, test_size=0.2, random_state=0)
svc = svm.SVC(kernel='linear').fit(X_train, y_train)
lin_svc = svm.SVC(kernel='linear').fit(X_train, y_train)
rbf_svc = svm.SVC(kernel='rbf').fit(X_train, y_train)
poly_svc = svm.SVC(kernel='poly', degree=3).fit(X_train, y_train)


# 建立網格座標系統(petal花瓣）
h = .02
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# the title of the graph
titles = ['SVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']


for i, clf in enumerate(( lin_svc, rbf_svc, poly_svc)):
    plt.subplot(2, 2, i + 1) 
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) 
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8) 
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()

lin_svc_pre = lin_svc.predict(X_test)
acc_lin_svc = sum(lin_svc_pre==y_test)/len(y_test)
rbf_svc_pre = rbf_svc.predict(X_test)
acc_rbf_svc = sum(rbf_svc_pre==y_test)/len(y_test)
poly_svc_pre = poly_svc.predict(X_test)
acc_poly_svc = sum(poly_svc_pre==y_test)/len(y_test)
print(acc_lin_svc)
print(acc_rbf_svc)
print(acc_poly_svc)