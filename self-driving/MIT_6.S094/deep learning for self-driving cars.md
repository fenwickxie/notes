# Deep Learning for Self-Driving Cars

+ deep raffic: multi agent deep reinforcement learning
+ segfuse: dynamic driving scene segmentation
  + **space and time**
    + 对场景中的空间视觉特征进行解释，并且理解并追踪场景的时间动态特征
    + 场景分割 + 信息的时间传播
+ deep crash: deep reinforcement learning for high-speed crash avoidance

## deep learning

```mermaid
erDiagram
    Artificial_Intelligence ||--|{ Machine_Learning : contains
    Machine_Learning  ||--|{ Representation(Feature)_Learning : contains
    Representation(Feature)_Learning ||--|{ Deep_Learning : contains
```

+ 人工神经网络是分层的、有序的、同步的，生物神经网络是混合的、无序的、异步的

+ **空间特征不一定能体现世界上物体的真正构建层次**

+ challenge
  + transfer learning
  + require big data
  + require supervised data
  + not fully automated， human involve needed
  + reward function
  + transparence
  + corner/edge case
