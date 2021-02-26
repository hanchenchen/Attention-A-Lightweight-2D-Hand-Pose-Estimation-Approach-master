# Attention-A-Lightweight-2D-Hand-Pose-Estimation-Approach-master TEST
#### 任务描述

在手部关键点检测任务中，对论文 **Attention! A Lightweight 2D Hand Pose Estimation Approach**  中提出的Attention Augmented Inverted Bottleneck Block等结构进行测试。

#### 测评环境

- Ubuntu 16.04.6 LTS
- Python 3.8.5 
- tensorflow 2.4.1

#### 数据集

| [CMU Panoptic][ http://domedb.perception.cs.cmu.edu/handdb.html] | [SHP][https://sites.google.com/site/zhjw1988/] | [FreiHAND Dataset][https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html] | [HO3D_v2][https://cloud.tugraz.at/index.php/s/9HQF57FHEQxkdcz?] |
| ------------------------------------------------------------ | ---------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 31836                                                        | 36000                                          | 130240                                                       | 66034                                                        |

所有图片剪裁为224*224大小，部分数据集的3D关键点标注投影为2D。

训练样本：验证样本：测试样本 = 80%：10%：10%

#### 测评指标

**PCK**：*Probability of Correct Keypoint within a Nor- malized Distance Threshold*

#### 测试大纲

使用**Convolutional Pose Machines**作为参考的基准，测试论文中提出的architecture是否有良好的性能

采用消融实验方法，对论文中使用的 Attention Augmented Inverted Bottleneck Block、Blur (Pooling Method)、Mish(Activation Function)进行测试。

##### 在原项目的基础上添加如下代码：

train.py: 增加了parser和json配置文件，便于在多个数据库上进行训练。

evaluate.py: 使用PCK指标对模型进行量的测试和质的测试。

（dataset_path）/crop_images.py: 将不同数据集中的图片剪裁为特定大小（224），并对labels进行修改

（dataset_path）/make_tfrecord.py: 将不同的数据集制作为tfrecord文件

model_ablation.py + arch.json: 实现了 IV. EVALUATION - B. Ablation studies 中的12种 architectures

model_cpm：使用Convolutional Pose Machines作为基准。

pck.py: 计算PCK。reference：NSRMhand-master[code][https://github.com/HowieMa/NSRMhand]

##### Train

```
python train.py (datatset_name) --arch (1-12/cpm) --GPU 0
```

```
python train.py HO3D_v2 --arch 1 --GPU 0
```

##### Evaluate

```
python evaluate.py (datatset_name) --arch (1-12/cpm) --GPU 0
```



#### 测试结果

见文件夹qualitative_results、quantitative_results。

模型权重文件：(dataset_name)/(arch_name )/weights.hdf5
当前完成 HO3D_v2：arch1 arch2 arch4

~~https://www.dropbox.com/sh/99u7apw2q52mzn2/AAD0JAmOQ8P4ZK-8VDXDR6xqa?dl=0~~

Architecture1 在不同数据集上的表现

（图1  pck）

在HO3D_v2数据集上，CPM与Architecture1进行比较

（图2 pck） + 模型大小比较

在HO3D_v2数据集上，消融实验，arch1、2、3、4，两个属性：Attention module、Pooling Method



##### Reference

Attention! A Lightweight 2D Hand Pose Estimation Approach [[code]][https://github.com/nsantavas/Attention-A-Lightweight-2D-Hand-Pose-Estimation-Approach]

对论文Attention-A-Lightweight-2D-Hand-Pose-Estimation-Approach-master[code][https://github.com/nsantavas/Attention-A-Lightweight-2D-Hand-Pose-Estimation-Approach] 进行测试和Pytorch复现。