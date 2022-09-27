# open-set-loss
- backbone : mobilenetV2  
- Loss : Entropic Open-set Loss

&#160;

## Environment
- python 3.8
- torch, torchvision, torchasudio installed according to the CUDA you are using
```
pip install -r requirements.txt
```

&#160;

## Dataset structure
<img src="./img/image1.PNG" width="100" height="100">  

&#160;
User needs to collect the images by class as shown in the picture.

&#160;

<img src="./img/image2.PNG" width="100" height="100">

&#160; 
Images are included as shown.

&#160; 

## How to train
```
python3 train.py --train_dataset_root {train_path} --validation_dataset_root {val_path} 
```

&#160;

## How to eval & export ONNX file
```
python3 test.py --dataset_root {test_path} --weight {weight_path} --export
```
- If the `export` option is given, an onnx file is created.

&#160;

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fys-jo%2Fopen-set-loss&count_bg=%233D46C8&title_bg=%23848E00&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)