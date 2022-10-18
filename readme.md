## 图像识别
数据集：ImageNet

模型：ResNet-50


### 输入参数说明
```shell
optional arguments:

  --image(str) 图片的路径 

  --method(str) {identify,fgsm,ifgsm,mifgsm,deefool} 识别或攻击方法
  
  --eps(float) 生成对抗样本时其扰动大小限制，范围 0-20

  --interation(int) 生成对抗样本时其迭代次数，范围 0-10

  --alpha(float) 生成对抗样本时其迭代步长，范围 0-2

  --decay(float) MI-FGSM攻击中的decay系数，范围 0-1 
```

### 输出参数说明
```shell
# 执行成功：
# 返回code字段固定为0, msg字段固定为success，data字段为具体信息
{'code': 0, 'msg': 'success', 'data': {...}}

# 执行失败
# 返回code字段固定为1, msg字段为栈错误信息，data字段为空
{'code': 1, 'msg': 'err msg', 'data': {}}
```

### 例子
#### 1. 识别图片
```shell
# 要且只要命令行中使用到的参数
python3 handler.py --image images/dog.jpg --method identify 

# 返回值
{
  'code': 0, 
  'msg': 'success', 
  'data': {
    'predict': 'sports car, sport car', 
    'distribution': {
      'sports car, sport car': '35.5%', 
      'convertible': '16.3%', 
      'car wheel': '5.9%', 
      'beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon': '4.1%',
       'grille, radiator grille': '2.1%'
    }
  }
}


```

#### 2. fgsm 攻击
```shell script
# 要且只要命令行中使用到的参数
python handler.py --image images/dog.jpg --method fgsm --eps 20

# 返回值（所有攻击的返回值格式一致）
{
  'code': 0, 
  'msg': 'success', 
  'data': {
    'predict': 'jeep, landrover',   # 预测值
    'distribution': {               # top5 预测分布
      'jeep, landrover': '14.1%',   
      'beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon': '6.9%', 
      'pickup, pickup truck': '4.2%', 
      'car wheel': '3.5%', 
      'Model T': '3.1%'
      }, 
    'adv_image_path': 'images/adv_car.jpg',     # 生成对抗样本图片所保存路径，保存路径与输入图片一致
    'noise_image_path': 'images/noise_car.jpg'  # 生成对抗噪声图片所保存路径，保存路径与输入图片一致
  }
}
```

#### 2. ifgsm 攻击
```shell script
# 要且只要命令行中使用到的参数
python handler.py --image 5.jpg --method ifgsm --eps 15 --iteration 10 --alpha 2

# 返回值（所有攻击的返回值格式与fgsm攻击一致）
```

#### 3. mifgsm 攻击
```shell script
# 要且只要命令行中使用到的参数
python handler.py --image 5.jpg --method mifgsm --eps 15 --iteration 10 --alpha 2 --decay 0.8

# 返回值（所有攻击的返回值格式与fgsm攻击一致）
```

#### 4. deepfool 攻击
```shell script
# 要且只要命令行中使用到的参数
python handler.py --image 5.jpg --method deepfool --eps 10 --iteration 1

# 返回值（所有攻击的返回值格式与fgsm攻击一致）
```