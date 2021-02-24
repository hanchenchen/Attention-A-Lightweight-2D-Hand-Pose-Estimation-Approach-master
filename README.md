# Attention-A-Lightweight-2D-Hand-Pose-Estimation-Approach-master TEST
#### 任务描述

在手部关键点检测任务中，对论文 **Attention! A Lightweight 2D Hand Pose Estimation Approach**  中提出的Attention Augmented Inverted Bottleneck Block等结构进行测试。

#### 测评环境

- Ubuntu 16.04.6 LTS
- Python 3.8.5 
- tensorflow 2.2.0

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

#### 测试结果





##### Reference

Attention! A Lightweight 2D Hand Pose Estimation Approach [[code]][https://github.com/nsantavas/Attention-A-Lightweight-2D-Hand-Pose-Estimation-Approach]



对论文Attention-A-Lightweight-2D-Hand-Pose-Estimation-Approach-master[code][https://github.com/nsantavas/Attention-A-Lightweight-2D-Hand-Pose-Estimation-Approach] 进行测试和Pytorch复现。

##### 在原项目的基础上添加如下代码：

model_train.py: 增加了parser和json配置文件，便于在多个数据库上进行训练。

（dataset_path）/crop_images.py: 将不同数据集中的图片剪裁为特定大小（224），并对labels进行修改

（dataset_path）/make_tfrecord.py: 将不同的数据集制作为tfrecord文件

model_ablation.py + arch.json: 实现了 IV. EVALUATION - B. Ablation studies 中的12种 architectures

model_evaluate.py: 使用PCK指标对模型进行测试

pck.py: 计算PCK。reference：NSRMhand-master[code][https://github.com/HowieMa/NSRMhand]

截至 2021/2/13 5pm 的训练结果如下:
GPU: Nvidia TITAN V, platform: TensorFlow, Dataset: FreiHAND_v2, Epoch: 196

PCK_results: {"0.0": 0.0, "0.05": 0.08999999999999998, "0.1": 0.28857142857142837, "0.15000000000000002": 0.5090476190476191, "0.2": 0.6961904761904759, "0.25": 0.7995238095238091, "0.30000000000000004": 0.8757142857142856, "0.35000000000000003": 0.93095238095238, "0.4": 0.9580952380952374, "0.45": 0.9757142857142855, "0.5": 0.9866666666666665, "0.55": 0.9909523809523807, "0.6000000000000001": 0.9947619047619046, "0.65": 0.9957142857142856, "0.7000000000000001": 0.9976190476190476, "0.75": 0.999047619047619, "0.8": 0.9995238095238095, "0.8500000000000001": 0.9995238095238095, "0.9": 0.9995238095238095}

GPU: GeForce RTX 3090, platform: TensorFlow, Dataset: HO3D_v2, Epoch: 35
PCK_results: {"0.0": 0.0, "0.05": 0.04761904761904754, "0.1": 0.0842857142857142, "0.15000000000000002": 0.2823809523809519, "0.2": 0.44238095238095204, "0.25": 0.6314285714285713, "0.30000000000000004": 0.7619047619047622, "0.35000000000000003": 0.8033333333333315, "0.4": 0.8290476190476174, "0.45": 0.8566666666666657, "0.5": 0.9052380952380945, "0.55": 0.9685714285714281, "0.6000000000000001": 0.999047619047619, "0.65": 1.0, "0.7000000000000001": 1.0, "0.75": 1.0, "0.8": 1.0, "0.8500000000000001": 1.0, "0.9": 1.0}