import os
import torch
from torchvision import  transforms
import argparse
from dataset import CustomDataset
from model import mobilenet_v2
from accuracy import Accuracy
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
    parser.add_argument('--input_size', default=[192,192], type=int, nargs=2,
                        help='input size (width, height)')
    parser.add_argument('--no_background', default=False, action='store_true',
                        help='Use background dataset')
    parser.add_argument('--export', default=False, 
                        action='store_true',
                        help='export onnx')
    parser.add_argument('--topk', type=int,
                        default=1,
                        help='topk...')
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

    x = torch.rand(1, 3, h, w)

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
        #rgbtor(),
        transforms.ToTensor(),
        #transforms.Normalize(0.5,0.5)]
        transforms.Normalize(args.mean, args.std)]
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

    metric = Accuracy(topk=args.topk, remove = len(class_names)-1, no_background = args.no_background)
    metric.reset()
    topk = args.topk
    remainder = len(test_dataset)%args.batch_size

    #?????? ??????
    model.eval()    # ??????????????? dropout??? OFF ??????.
    correct = 0
    background_correct_num = 0
    wrong = 0
    background_num = 0
    total = 0
    if not args.no_background:
        class_correct = list(0. for i in range(len(class_names) - 1))
        class_total = list(0. for i in range(len(class_names) - 1))
    else:
        class_correct = list(0. for i in range(len(class_names)))
        class_total = list(0. for i in range(len(class_names)))        
    for data, target in test_loader:
        if torch.cuda.is_available():
            data = data.to(device)
            target = target.to(device)
        output = model(data)
        pred = torch.exp(output)
        top_prob, top_class = pred.topk(topk, 1)
        target_ = target.unsqueeze(1).expand_as(top_class)
        c = (top_class == target_).squeeze()
        #prediction = torch.max(output[0])
        if (len(target) < args.batch_size):
            for i in remainder:
                if args.no_background is False:
                    if torch.max(output[i]) < 0.5 and target[i] == len(class_names)-1:
                        correct += 1
                        background_num +=1
                        background_correct_num +=1
                    elif target[i] == len(class_names)-1:
                        wrong +=1 
                        background_num += 1      
                    else:
                        if torch.argmax(output[i]).eq(target[i]):
                            correct += torch.argmax(output[i]).eq(target[i])
                            class_correct[target[i]] += torch.argmax(output[i]).eq(target[i])
                        else:
                            wrong += 1
                    class_total[target[i]] += 1
                    total += 1
                else:
                    correct += torch.argmax(output[i]).eq(target[i])
                    class_total[target[i]] += 1
                    class_correct[target[i]] += torch.argmax(output[i]).eq(target[i])
                    total += 1
        else:
            for i in range(args.batch_size):
                if args.no_background is False:
                    if torch.max(output[i]) < 0.5 and target[i] == len(class_names)-1:
                        correct += 1
                        background_num +=1
                        background_correct_num +=1
                    elif target[i] == len(class_names)-1:
                        wrong +=1 
                        background_num += 1      
                    else:
                        if torch.argmax(output[i]).eq(target[i]):
                            correct += torch.argmax(output[i]).eq(target[i])
                            class_correct[target[i]] += torch.argmax(output[i]).eq(target[i])
                        else:
                            wrong += 1
                        class_total[target[i]] += 1
                    total += 1
                else:
                    correct += torch.argmax(output[i]).eq(target[i])
                    class_total[target[i]] += 1
                    class_correct[target[i]] += torch.argmax(output[i]).eq(target[i])
                    total += 1
        metric.match(model(data), target)
    accuracy = metric.get_result()
    # print results
    print("==================================================")
    print("========??????????????? ?????? ????????? ??????===============")
    print("accuracy (without background class) = %.3f %%" % accuracy)
    for i in range(len(class_names)-1):
        print('Accuracy of %s: %.3f %%' % (class_names[i], 100 * class_correct[i] / class_total[i]))

    print("==================================================")
    print("========??????????????? ?????? ????????? ??????===============")
    if args.no_background is False:
        print("total_wrong: ", wrong)
        print("total_correct: ", int(correct.detach().cpu()))
        print("Background num: ", background_num)
        print("Background correct num: ",background_correct_num)
        print('Test set Accuracy without background : {:.2f}%'.format(100. * (correct-background_correct_num) / (total-background_num)))
    print('Test set Accuracy : {:.2f}%'.format(100. * correct / total))
