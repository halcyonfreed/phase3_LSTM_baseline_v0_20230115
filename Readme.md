标了“可改”的地方可以改
每次data文件夹删除了以后再上传，复制过来以后再调试


## 任务说明

1. 已知：global_x,y是在整个大区里面的坐标，超级大学它没意义
2. local_x,y：记录了一段区域，从某一个起点然后记录比如600m的高速路段，这个还是太大学它没意义，应该学相对于自车为原点(0,0)的(Δx,Δy)

## 数据说明

### trainset

1. trainset：

   trainset=scipy.io.loadmat('x/mat')

   1. dict
   2. len：5
   3. dict_keys(['__header__', '__version__', '__globals__', 'tracks', 'traj']) 只用看最后两个有用，前面是matlab内置说明信息

2. trainset['tracks']:

   1. type: <class 'numpy.ndarray'> 

   2. shape:  (6, 2356)

   3. 6: dataset有us101 i80一共6个，是最外层的ndarray，里面嵌套了长度是2356的一个ndarray

   4. 2356：每一列又是一个ndarray最内层，但是二维的矩阵长宽不定变长，但是长*宽最大=2356

      所以有的不够长补成空的[]维度是[1,0]，有[4,429]各种长度

3. trainset['traj']:
   1. type: <class 'numpy.ndarray'> 

   2. shape:  (5922867, 47) 

   3. 5922867：车的辆数*时间的帧数

   4. 47：47个特征feature：

      ​	都是float格式，后期要.astype(int)

      1. 0: datasetID
      2. 1: vehID
      3. 2: frame_ID 帧数标号
      4. 3: local_x
      5. 4: local_y
      6. 5: lane_ID
      7. 6: lateral_class 直1 左2 右3
      8. 7: longitudinal_class 刹车2 不刹车1
      9. 8-46: 3*13格子，每格存vehID只能一辆车

### utils

1. getHist
   1. type: array
   2.  shape=(0, 2),
   3. dtype=float64

## 学习

### debug

1. 在ipynb里面调试，然后在py里面写到主程序里
2. _,__,a , _in b: 说明b里面四个变量只取一个a
3. 粗略的讲，batch_size管显存，num_workers管GPU利用率。 batch_size设置一般为2-32，num_workers一般为**8、16**

### numpy

1. A.astype(数据类型) ：A是ndarray或者pandas的series class，然后is a method within numpy.ndarray, as well as the Pandas Series class

2. ndarray取成员可以[a][b]也可以[a,b]

3. np.eye([2]) 和np.eye(2)一样

   np.eye默认是float所以astype(int)

4. np.zeros()

5. b=a[np.where(a[:,0==t])][0,1:3]

6. [行,列]

7. a.shape: [4,4]; a.size:16

8. np.where(条件) np.argwhere(条件) 返回tuple



### scipy

1. scipy.io.loadmat('x/mat')返回dict每个成员是ndarray

### torch

1. byte(): 转dtype从FloatTensor64到torch.uint8 64浮点数到8整型
2. torch.from_numpy()

