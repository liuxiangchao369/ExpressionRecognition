
# 数据集下载

https://paperswithcode.com/dataset/fer2013

# 网络参数量

```commandline
Total params: 876,887
Trainable params: 876,887
Non-trainable params: 0
Total mult-adds (M): 596.85
==========================================================================================
Input size (MB): 0.02
Forward/backward pass size (MB): 26.50
Params size (MB): 3.35
Estimated Total Size (MB): 29.86
```


# 使用额外数据集

## 训练超参数



随机种子: 866689041

batch_size=64

lr=1e-05

max_epochs=120

学习率衰减 scheduler = StepLR(optimizer, step_size=60, gamma=0.5)  





## 性能指标

Accuracy: 0.990

| expression |   pre        |      rec        |      f1-score      |    num_sample |
|----------|----------|----------|----------|----------|
|angry     |  0.9906    |        0.9896    |         0.9901      |       958       |
|disgust   |  0.9910    |        0.9910     |        0.9910      |       111      | 
|fear      |  0.9931     |       0.9834     |        0.9882      |       1024     | 
|happy     |  0.9966    |        0.9977     |        0.9972      |       1774      |
|neutral    | 0.9785     |       0.9976     |        0.9880      |       1233     | 
|sad        | 0.9927     |       0.9800     |        0.9863      |       1247     | 
|surprise   | 0.9904     |       0.9916     |        0.9910      |       831      | 

![混淆矩阵](Confusion%20Matrix.jpg)



# 仅使用FER-2013数据集

## 训练超参数

随机种子: 2134640168

batch_size=64

lr=1e-5

max_epochs=200

学习率衰减 scheduler = StepLR(optimizer, step_size=50, gamma=0.5)  

## 性能指标

Accuracy: 0.9789634995820563

| expression |   pre        |      rec        |      f1-score      |    num_sample |
|----------|----------|----------|----------|----------|
|angry    |   0.9914     |       0.9614     |        0.9762      |       958       
|disgust  |   0.9907     |       0.9640      |       0.9772      |       111       
|fear      |  0.9427      |      0.9805      |       0.9612      |       1024      
|happy    |   0.9955      |      0.9966       |      0.9961       |      1774      
|neutral  |   0.9844     |       0.9724      |       0.9784       |      1233      
|sad      |   0.9613      |      0.9759      |       0.9686       |      1247      
|surprise  |  0.9939      |      0.9759      |       0.9848       |      831 

![混淆矩阵](Confusion%20Matrix2.jpg)


# 快速开始

```commandline
python3 emotion_det.py --img_path="path to your image"
```

# 从头构建

1.训练模型

```commandline
python3 train.py
tensorboard --logdir log
```
![loss](loss.jpg)
2.pytorch模型转为onnx

```commandline
python3 torch2onnx.py
```

3.测试
```commandline
python3 test.py
```