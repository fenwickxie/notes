<style>@import url(../../css/auto-number-title.css); </style>

# BEV(Bird’s-Eye-View)

一种鸟瞰视图的传感器数据表示方法

## BEV感知算法简介
### BEV感知算法的概念    
#### BEV   
  + Bird’s-Eye-View，鸟瞰图（俯视图）   
  + 尺度变化小  
    + 网络对特征一致的目标表达能力更好  
  + 遮挡小  
#### 感知  
  + 一种响应模式，系统对外界的响应    

```mermaid
---
config:
  theme: default
  themeVariables:
    fontFamily: 'Times New Roman'
    fontSize: 14px
---
block-beta
columns 8
  classDef tag fill:#ffffff,stroke:#ffffff,stroke-width:1px,color:#000000,bold:true;
  classDef sub fill:#ccd8f3,stroke:#333333;
  classDef core fill:#abc2f1,stroke:#333333;
  classDef basic fill:#f1f1f1,stroke:#c5c5c5;
  classDef input fill:#e6e6e6,stroke:#ffffff;

  id0>"核心任务"]:1  id01["基础模型"]:1  id02["BEV感知"]:3 id03["三维重建"]:1  id04["联合感知/决策"]:2
  space:3
  blockArrowId1<["&nbsp;&nbsp;&nbsp;"]>(up)
  space:5
  id11["透视感知"]:1
  block:group3:3
    columns 3
    id12["BEV相机"]:1 id13["BEV融合"]:1 id14["BEV激光雷达"]:1 
  end
  space:6
  blockArrowId2<["&nbsp;&nbsp;&nbsp;"]>(up)
  space:4
  id3>"基础任务"]:1
  
  block:group1:5
    columns 5
    id21["感知智能"]:5   
    id31["分类"]:1 id32["检测"]:1 id33["分割"]:1 id34["跟踪"]:1 id35["预测"]:1 
  end

  block:group2:2
    columns 2
    id22["决策智能"]:2
    id36["规划"]:1 id37["控制"]:1
  end

  id4>"输入数据"]:1 id41["视觉/相机"]:1 id42["激光雷达"]:1 id43["毫米波雷达"]:1 id44["GNSS/GPS"]:1 id45["里程计/运动学"]:1 id46["高精地图"]:1 id47["CAN总线"]:1

  group1 -->id04
  group2 -->id04
  id41 --> id12
  id41 --> id13
  id42 --> id14
  id42 --> id13

  class id0,id21,id22,id3,id21,id22,id4 tag
  class id11,id12,id13,id14 sub
  class id31,id32,id33,id34,id35,id36,id37 basic
  class id41,id42,id43,id44,id45,id46,id47 input
  class id01,id02,id03,id04 core
```

#### BEV感知 
  + 建立在众多子任务上的一个概念  
  + 包括分类、检测、分割等  
#### BEV感知输入  
  + 包括毫米波雷达、激光点云雷达、相机图像等  
  + 根据输入不同有进一步划分不同类型BEV感知算法  
### BEV感知算法的数据形式    
#### 纯图像  
> + 三维世界映射到二维像素表示  
> + 纹理丰富、成本低  
> + 基于图像的任务、基础模型相对成熟，易扩展到BEV  
> *涉及图像的方法使用的基本是图像处理框架中的一些通用网络，如resnet等*
+ BEVFormer  
<p align='center'><img src='asserts/bevformer.png'></p>

#### 纯点云  
> + 稀疏性  
> + 无序性  
> + 3D表征  
+ 点云特征提取方法——采用一定的聚合方法，将点云数据聚合为特征图  
  + 基于点的（point-based）：聚合关键点和其周围（一个球体空间内）点  
  + 基于体素的（voxel-based）：聚合一定区域内的点  
#### 图像 + 点云  
+ BEVFusion  
<p align='center'><img src='asserts/bevfusion.png'></p>

### BEV开源数据集  
#### [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/)  
+ 转换矩阵
  + $$ y=P_{rect}^{(i)}R_{rect}^{(0)}T_{velo}^{cam}x$$  
    + $T$——激光雷达坐标系转换到相机坐标系  
    + $R$——相机畸变矫正矩阵  
    + $P$——相机内参矩阵，3D到2D映射  
  
<p align='center'><img src='asserts/kitti_annotation.png' width=50%></p>  

+ 标注文件  
  + 按场景标注  
    + 对每个场景进行编号，并有一个同名标注文件  
  + 单个标注文件  
    + 每行表示一个物体  
  + 单行  
    + 目标类型 $class$  
    + 目标被截断程度 $cut\in[0,1]$  
    + 目标被遮挡程度 $obstruction \in\{0,1,2,3\}$，离散值  
    + 目标与相机之间的夹角 $\theta\in[-\pi,\pi]$   
    + 目标边界框左上角和右下角坐标 $(x_{left},y_{left},x_{tight},y_{right})$  
    + 目标的3D尺寸 $(h,w,l)$ ，单位m 
    + 目标在3D场景下的中心点坐标 $(x_{c},y_{c},z_{c})$  ，单位m  
    + 目标在此位置以此类别存在的概率，即置信度得分 $score \in [0,1]$ 
#### [**<font color=red>nuScenes</font>**](https://www.nuscenes.org/)  
> + maps：  栅格化图像和
> + samples：**关键帧**传感器数据，已标注的图像  
> + sweeps：  **中间帧**传感器数据，未标注的图像  
> + v1.0-*：元数据、标注数据  
>   + attribute.json：实例属性  
>   + calibrated_sensors.json：传感器（激光雷达/相机）标定数据  
>   + category.json：对象类别  
>   + ego_pose.json：车辆特定时刻的姿态  
>   + instance.json：一个物体的实例  
>   + log.json：日志信息  
>   + map.json：二值分割掩模地图信息  
>   + sample.json：样例  
>   + sample_annotation.json：3D边界框  
>   + sample_data.json：传感器数据
>   + scene.json：场景数据  
>   + sensor.json：传感器信息  
>   + visibility.json：实例可见性

### BEV感知方法分类  
#### BEV Lidar  
```mermaid
graph LR

A[点云输入] --> B[体素化] & C[转换为BEV]

subgraph Pre-BEV
  B --> D[3D特征提取]-->E[转换为BEV]
end

subgraph Post-BEV
  C ---> F[2D特征提取]
end

E &　F-->G[检测头]
```

+ Pre-BEV feature extraction  
  先提取特征，再生成BEV表征
  + PV-RCNN  
    + <p align='center'><img src='asserts/pv-rcnn.png'></p> 
    +  point + voxel --> BEV feature map
+ Post-BEV feature extraction  
  先转换到BEV视图，再提取特征
  + PointPillar  

#### BEV Camera  
```mermaid
graph LR

A[多视角图像] --> B[共享2D特征提取模块] 

subgraph 视角转换模块
  direction TB
  C[2D-3D]
  D[3D-2D]
  C ~~~ D
end

B--> 视角转换模块-->E[3D解码器检测头]
```

+ BEVFormer  
  + <p align='center'><img src='asserts/bevformer.png'></p>

#### BEV Fusion  
> **融合是在特征层面的融合**  

+ BEVFusion  
  + <p align='center'><img src='asserts/bevfusion.png'></p>

### BEV感知算法的优劣  

```mermaid
graph TB

A([多视角图像]) 
B([点云]) 
A-->C[图像视图算法]--2D结果-->D[2D-3D转换]--3D结果-->T[时间和空间]
B--> L[激光雷达网络]--3D结果--->E[时间和空间]--> F[融合]-->P([感知结果])
T-->F

A1([多视角图像]) 
B1([点云]) 
A1-->C1[特征提取器]--PV特征-->D1[2D-3D转换]
B1--> L1[特征提取器]--BEV特征---> F1[融合]-->E1[时间和空间]-->P1([感知结果])
D1--BEV特征-->F1
```

+ BEV感知算法对学术研究的意义  
  + 利于探讨2D到3D的转换过程  
  + 利于视觉图像识别远距离物体或颜色引导的道路  
+ BEV感知算法对工业应用的意义  
  + 降低成本，激光雷达设备成本是视觉设备的10倍  
+ 性能差异  
  + BEV感知算法在感知距离上优于2D感知算法3D检测任务上与点云方案还有一定差距  

### BEV感知算法的应用  
+ Tesla  

```mermaid
---
config:
  theme: default
  themeVariables:
    fontFamily: 'Times New Roman'
    fontSize: 14px
---
block-beta 
columns 3
  classDef tag fill:#ffffff,stroke:#ffffff;
  classDef basic fill:#f1f1f1,stroke:#c5c5c5;
  classDef arrow fill:#ffffe4,stroke:#ffffe4;

  i00["原图"]:1 i01["原图"]:1 i02["原图"]:1
  i10["矫正"]:1 i11["矫正"]:1 i12["矫正"]:1
  i20["RegNET"]:1 i21["RegNET"]:1 i22["RegNET"]:1
  i30["BiFPN"]:1 i31["BiFPN"]:1 i32["BiFPN"]:1
  i40["多尺度特征"]:1 i41["多尺度特征"]:1 i42["多尺度特征"]:1
  
  block:id5:3
    columns 4
    i50["多相机融合&BEV变换"]:1
    i510["PV特征"]:1 space:1 i512["BEV特征"]:1
    i510--"变换"-->i512
  end
  i60(["IMU"]):1

  block:i61:2
    columns 2
    i610["特征序列"]:1
    block:i611:1
      columns 11
      a1["&nbsp;"] space l["&nbsp;"] m["&nbsp;"] n["&nbsp;"] o["&nbsp;"] p["&nbsp;"] q["&nbsp;"] r["&nbsp;"] space a2["&nbsp;"]
      a1-->l
      r-->a2
    end
  end

  i60-->i61
  i7["视频模块"]:3
  i801("解码器"):1
  i802("解码器"):2
  block:id81:1
    columns 2
    i810["分类"]:1 i812["回归"]:1
  end
  block:i82:2
    columns 4
    i820["分类"]:1 i822["回归"]:1 i823["属性"]:1
    
  end
  class i50 tag
  class a1,a2,i50 arrow
  class i801,i802 basic
```



+ Horizon Robotics  

```mermaid
---
config:
  theme: default
  themeVariables:
    fontFamily: 'Times New Roman'
    fontSize: 14px
---
block-beta 
columns 5
  classDef task fill:#f1f1f1,stroke:#000000,stroke-width:1px,color:#000000,stroke-dasharray: 5 5;
  
  i00["原图"]:1 space i01["点云"]:1 space i02["IMU\GPS"]:1
  i10["单相机前端"]:1 space i11["激光雷达前端"]:1 space i12["其他传感器前端"]:1
  i20["交叉流对齐"]:1 space i21["交叉模态对齐"]:1 space i22["学习的时空聚合"]:1
  i20--"2D-3D"-->i21
  i21-->i22
  space:5
  i30["底层物理学"]:1 i31["语义层实体提取"]:1 i32["结构层概念,关系,行为"]:1 space:2
  i20-->i30
  i21-->i31
  i21-->i32
  space:5
  i40["视差/深度/光流..."]:1 i41["检测:行人/车辆/道路"]:1 i42["跟踪/预测..."]:1 space:2
  i30-->i40
  i31-->i41
  i32-->i42
  class i40,i41,i42 task
```

+ HAOMO  

```mermaid
---
config:
  theme: default
  themeVariables:
    fontFamily: 'Times New Roman'
    fontSize: 14px
---
block-beta 
columns 3
  classDef tag fill:#ffffff,stroke:#ffffff;
  classDef basic fill:#f1f1f1,stroke:#c5c5c5;
  classDef arrow fill:#ffffe4,stroke:#ffffe4;

  i00["点云"]:1 i01["原图1"]:1 i02["原图2"]:1
  i10["Pillar特征网络"]:1 i11["ResNet"]:1 i12["ResNet"]:1
  i20["CNN主干网络"]:1 i21["FPN"]:1 i22["FPN"]:1
  i30["BEV特征"]:1 i31["多尺度特征"]:1 i32["多尺度特征"]:1
  blockArrowId6<["&nbsp;"]>(down):1
  block:i4:2
    columns 4
    i40["transformer"]:1
    i400["PV特征"]:1 space:1 i401["BEV特征"]:1
    i400--"变换"-->i401
  end
  i5("张量空间"):3
  
  block:i6:3
    columns 3
    i60["特征序列"]:1
    space
    block:i61:1
      columns 11
      a1["&nbsp;"] space l["&nbsp;"] m["&nbsp;"] n["&nbsp;"] o["&nbsp;"] p["&nbsp;"] q["&nbsp;"] r["&nbsp;"] space a2["&nbsp;"]
      a1-->l
      r-->a2
    end
  end

  block:i7:3
    columns 3
    i70["时空融合"]:1
    i700["RNN/Transformer"]:1 i701["SLAM光流追踪"]:1
  end

  i801("解码器"):1
  i802("解码器"):2
  block:id81:1
    columns 2
    i810["分类"]:1 i812["回归"]:1
  end
  block:i82:2
    columns 4
    i820["分类"]:1 i822["回归"]:1 i823["属性"]:1 
  end
  class a1,a2,i40,i60,i70 arrow
  class i801,i802 basic
```

## BEV感知算法基础模块  
### 2D图像处理  

```mermaid
graph LR
  i0[多视角图像]-->i1[主干网络]-->i3[多视角输出]
```
### 3D点云特征处理  

```mermaid
graph LR
  i0["点云数据"]-->i1["基于点的(point-based)"] & i2["基于体素的(voxel-based)"]-->i3["输出"]
  
```
+ 基于点的(point-based)
  + <p align=center><img src='asserts/pointnet++.png'></p>
### 2D-3D  

### 3D-2D  

### BEV中的transformer  


## BEV融合感知算法  

## 基于环视camera的BEV感知算法  

## BEV实战  
