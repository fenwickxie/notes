<style> @import url(../auto_number_title.css); </style>

# Pytorch

## **数据获取及加载**

### **Dataset类**

提供获取数据及其标签（label）的方式；指示数据量的多少。

数据集组织形式：①数据文件名作为标签；②数据对应同名TXT文件中定义数据标签（类别、坐标等信息）。

自定义数据集，须重写__get_item__方法

### **Dataloader类**

数据加载，为网络提供不同结构格式的数据

## **数据记录及可视化**

### **torch.utils.tensorboard**

#### **SummaryWriter类**

输出log文件

##### **常用方法**

add_image(self, img_tensor, global_step: global step to record,wall_time)	添加图像数据

add_scalar(self, tag: data identifier, scalar: scalar value, global_step: global step to record,wall_time)	添加训练日志，如训练loss等信息

close()

## **图像变换**

### **torchvision.transforms**

1. Compse类——结合多种变换
2. ToTensor类——PIL image/ndarray转换为张量
3.
