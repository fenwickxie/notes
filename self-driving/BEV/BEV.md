<style>@import url(../../css/auto-number-title.css); </style>

# BEV(Bird’s-Eye-View)

一种鸟瞰视图的传感器数据表示方法

## BEV感知算法的概念    
+ BEV   
  + Bird’s-Eye-View，鸟瞰图（俯视图）   
  + 尺度变化小  
    + 网络对特征一致的目标表达能力更好  
  + 遮挡小  
+ 感知  
  + 一种响应模式，系统对外界的响应    

```mermaid
---
config:
  theme: default
  themeVariables:
    fontFamily: 'Times New Roman'
    fontSize: 14
---
block-beta
columns 8
  classDef tag fill:#ffffffff,stroke:#ffffff,stroke-width:1px,color:#000000,bold:true;
  classDef sub fill:#ccd8f3,stroke:#333333;
  classDef core fill:#abc2f1,stroke:#333333;
  classDef basic fill:#f1f1f1,stroke:#c5c5c5;
  classDef input fill:#e6e6e6,stroke:#ffffff;

  id0>"核心任务"]:1  id01["基础模型"]:1  id02["BEV感知"]:3 id03["三维重建"]:1  id04["联合感知/决策"]:2
  space:8
  space:1 id11["透视感知"]:1 id12["BEV相机"]:1 id13["BEV融合"]:1 id14["BEV激光雷达"]:1 space:3

  
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

  id4>"输入数据"]:1 id41["毫米波雷达"]:1 id42["激光雷达"]:1 id43["视觉/相机"]:1 id44["GNSS/GPS"]:1 id45["里程计/运动学"]:1 id46["高精地图"]:1 id47["CAN总线"]:1

  id12 -->id02 id13 -->id02 id14 -->id02

  class id0,id21,id22,id3,id21,id22,id4 tag
  class id11,id12,id13,id14 sub
  class id31,id32,id33,id34,id35,id36,id37 basic
  class id41,id42,id43,id44,id45,id46,id47 input
  class id01,id02,id03,id04 core
```

+ BEV感知 
  + 建立在众多子任务上的一个概念  
  + 包括分类、检测、分割等  
+ BEV感知输入  
  + 
## BEV感知算法的优势    
BEV感知算法的优势在于它能够提供更全面和更准确的周围环境信息，因为它将来自多个传感器的数据融合在一起，并且能够生成一个鸟瞰图视图，使得自动驾驶车辆能够更好地理解其周围环境，从而提高自动驾驶车辆的安全性和可靠性。   
