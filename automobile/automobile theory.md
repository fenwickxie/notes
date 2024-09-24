# 汽车理论

## 动力性

### 动力性指标

无风或微风条件
+ 最高车速$u_{max}$
  + 水平良好路面
+ 加速时间$t$
  + 原地起步加速时间：1、2档起步，适当换挡加速到预定车速或距离
    + 0-->400m
    + 0-->100km/h
  + 超车加速时间：最高档、次高档加速至某一高速
+ 最大爬坡度$i_{max}=tan\alpha$
  + 满载（某一载质量），良好路面
 
### 驱动力与行驶阻力

+ 驱动力<br>
作用于驱动轮的转矩：<br>
$$
T_t=T_{tq}i_{g}i_{0}\eta_{T}
T_{tq}：发动机转矩
i_{g}：变速器传动比
i_{0}：减速器传动比
\eta_{T}：传动系统机械效率
$$
驱动力：<br>
$$
F_t=\frac{T_{tq}i_{g}i_{0}\eta_{T}}{r}
r：车轮半径
$$

  + 发动机转速特性
    + 发动机外特性曲线:发动机节气门全开（或最大供油）
    + 发动机部分负荷特性曲线：发动机节气门部分开启（或部分供油）
        ![image](https://github.com/user-attachments/assets/0e2c67bd-8399-4161-9bc6-adad6286f3c0)
  
    + 发动机功率与转矩关系：<br>
      $$
      P_{e}=\frac{T_{tq}n}{9550}
      {T_{tq}：N\cdot m
      P_{e}：KW
      n：r/min
      $$<br>
        ![image](https://github.com/user-attachments/assets/38f48729-9e0f-4d62-83bf-03e3a6c66703)
  
  + 传动系统的机械效率
    + $$\eta_{T}=\frac{P_{in}-P_{T}}{P_{in}}$$
    + 等速行驶，$P_{in}=P_{e}$：$$\eta_{T}=1-\frac{P_{T}}{P_{e}}$$
    + 功率损失包括：
      + 机械损失：齿轮传动副、轴承、油封处的摩擦损失
        + 啮合齿轮对数
        + 传递的转矩
      + 液力损失：润滑油搅动、润滑油与旋转零件的表面摩擦
        + 润滑油品种
        + 温度
        + 箱体内油面高度
        + 齿轮等旋转零件的转速
      ![image](https://github.com/user-attachments/assets/babf8dd1-63ec-45d8-aac1-f38ca7192850)
  + 车轮半径
    + 车轮无载时半径为**自由半径**
    + 汽车静止时，车轮中心至轮胎与道路接触面间的距离为**静力半径$r_{S}$**
    + **滚动半径**：$r_{r}=\frac{S}{2\pi n_{w}}$,$n_{w}$为车轮转动的圈数；$S$为在转动$n_{w}$圈时车轮滚动的距离
      + 推荐按以下公式估算**滚动圆周**：$C_{R}=Fd$
        + $d$为自由直径
        + $F$为计算常数，子午线轮胎$F=3.05$，斜交轮胎$F=2.99$
  + 汽车驱动力图：驱动力与车速之间的函数关系式
    + $$u_{a}=0.377\frac{rn}{i_{g}i_{0}},r/min \rightarrow km/h=\frac{2\pi \times 60}{1000}=0.377$$

+ 行驶阻力<br>
$$\sum F=滚动阻力（F_{f}）+空气阻力（F_{w}）+坡度阻力（F_{i}）+加速阻力（F_{j}）$$
  + 滚动阻力
  + 空气阻力
    + 压力阻力
      + 形状阻力
      + 干扰阻力
      + 内循环阻力
      + 诱导阻力
    + 摩擦阻力
  + 坡度阻力
  + 加速阻力

+ 行驶方程式
  + $$F_{t}=F_{f}+F_{w}+F_{i}F_{j}\Longrightarrow \frac{T_{tq}i_{g}i_{0}\eta_{T}}{r}=Gf\cos \alpha + \frac{C_{D}A}{21.15}u_{a}^{2} + G\sin \alpha + \sigma m\frac{du}{dt}$$
  + 实际道路坡度不大，$\cos \alpha \approx 1, \sin \alpha \approx \tan \alpha$，上式写为：$$F_{t}=F_{f}+F_{w}+F_{i}F_{j}\Longrightarrow \frac{T_{tq}i_{g}i_{0}\eta_{T}}{r}=Gf + \frac{C_{D}A}{21.15}u_{a}^{2} + Gi + \sigma m\frac{du}{dt}$$





