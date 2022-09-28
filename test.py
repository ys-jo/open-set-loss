import os
import torch
from torchvision import  transforms
import argparse
from dataset import CustomDataset
from model import mobilenet_v2
import numpy as np
from PIL import Image
def parser():
    parser = argparse.ArgumentParser(description='evaluate network')
    parser.add_argument('--model', default='mobilenetv2',choices=['mobilenetv2'],
                        help='Detector model name')
    parser.add_argument('--dataset_root', default=None, type=str,
                        help='Dataset root directory path')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_workers', default=12, type=int,
                        help='Number of workers used in dataloading')  
    # model parameter
    parser.add_argument('--mean', nargs=3, type=float,
                        default=(0.486, 0.456, 0.406),
                        help='mean for normalizing')
    parser.add_argument('--std', nargs=3, type=float,
                        default=(0.229, 0.224, 0.225),
                        help='std for normalizing')
    parser.add_argument('--weight', default=None,
                        help='Weight file path')
    parser.add_argument('--input_size', default=[256,256], type=int, nargs=2,
                        help='input size (width, height)')
    parser.add_argument('--no_background', default=False, action='store_true',
                        help='Use background dataset')
    parser.add_argument('--export', default=False, 
                        action='store_true',
                        help='export onnx')
    args = parser.parse_args()
    return args


def build_model(args, custom_class_num=None):
    model = args.model.lower()

    if model.startswith('mobilenetv2'):
        return mobilenet_v2(custom_class_num=custom_class_num)

    raise Exception("unknown model %s" % args.model)


def load_model(model, source, optimizer=None, eval=0):
    if not os.path.isfile(source):
        raise Exception("can not open checkpoint %s" % source)
    checkpoint = torch.load(source)
    if eval==1:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        
    return epoch, loss


def prepare_model(args, class_numes):
    model = build_model(args, custom_class_num=class_numes)
    print("class num:", class_numes)

    _=load_model(model, source=args.weight, eval=1)

    if torch.cuda.is_available():
        model = model.cuda()

    return model


def export_classificaton_model(size, model):
    w = size[0]
    h = size[1]

    x = torch.rand(1, 1, h, w)

    if torch.cuda.is_available():
        x = x.cuda()
    filename = model.name + '.onnx'
    print('dumping network to %s' % filename)
    torch.onnx.export(model, x, filename)


#dataset
"""
dataset structure
train/
    class1/
        *.jpg
    class2/
        *.jpg
    class3/
        *.jpg
    background/
        *.jpg
test/
    class1/
        *.jpg
    class2/
        *.jpg
    class3/
        *.jpg
    background/
        *.jpg
"""
class rgbtor:
    def __init__(self):
        pass

    def __call__(self, img):
        r,g,b = img.split()

        return r

if __name__ == "__main__":
    args = parser()
    #use cuda
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')

    # load weight
    if not args.weight:
        raise Exception('You must enter weight path')

    # dataset_root check
    if not os.path.isdir(args.dataset_root):
        raise Exception("There is no dataset_root dir") 

    t = [transforms.Resize((args.input_size[1], args.input_size[0])),
        rgbtor(),
        transforms.ToTensor(),
        transforms.Normalize(0.5,0.5)]
        # transforms.Normalize(args.mean, args.std)]
    t = transforms.Compose(t)
    test_dataset = CustomDataset(data_set_path=args.dataset_root, transforms=t)
    class_names = os.walk(args.dataset_root).__next__()[1]
    class_names.sort()
    if "background" in class_names:
        class_names.remove("background")
        class_names.append("background")

    if args.no_background is False:
        if not "background" in class_names:
            raise Exception("There is no background dataset")

    # prepare model
    if args.no_background is False:
        model = prepare_model(args,class_numes = len(class_names)-1)
    else:
        model = prepare_model(args,class_numes = len(class_names))
    if args.export:
        export_classificaton_model(args.input_size, model)

    
    print('number of test data : ', len(test_dataset))
    # data loader
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size = args.batch_size, drop_last=True, shuffle = True, num_workers = args.num_workers)
   
    if torch.cuda.is_available():
        model = model.to(device)
        model = torch.nn.DataParallel(model)

    #모델 평가
    model.eval()    # 평가시에는 dropout이 OFF 된다.
    correct = 0
    no = 0
    wrong = 0
    for data, target in test_loader:
        if torch.cuda.is_available():
            data = data.to(device)
            target = target.to(device)
        output = model(data)
        prediction = torch.max(output[0])
        # print(prediction)
        # print("target", target)
        for i in range(args.batch_size):
            # print(output[i])
            # print(target[i])
            if args.no_background is False:
                if torch.max(output[i]) < 0.6 and target[i] == len(class_names)-1:
                    correct += 1
                    no +=1
                elif target[i] == len(class_names)-1:
                    wrong +=1 
                elif torch.max(output[i]) < 0.6:
                    pass
                else:
                    correct += torch.argmax(output[i]).eq(target[i])
            else:
                print(torch.argmax(output[i]))
                correct += torch.argmax(output[i]).eq(target[i])
                # correct += output[i].eq(target[i])

    if args.no_background is False:
        print("wrong: ", wrong)
        print("Background num: ",no)
    print('Test set Accuracy : {:.2f}%'.format(100. * correct / len(test_loader.dataset)))
    print('Test set Accuracy without background : {:.2f}%'.format(100. * (correct-no) / (len(test_loader.dataset)-wrong-no)))
