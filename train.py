import os
import torch
from torchvision import  transforms
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau,CosineAnnealingLR
import argparse
from torch import nn
from dataset import CustomDataset
from model import mobilenet_v2
from efficientnet import efficientnet_b0, efficientnet_b1
from loss import EntropicOpenSetLoss
import numpy as np
from PIL import Image
from adamp import SGDP
import torch.nn.functional as F



def parser():
    parser = argparse.ArgumentParser(description='Classification Training')

    parser.add_argument('--train_dataset_root', default=None,
                        help='train Dataset root directory path')
    parser.add_argument('--validation_dataset_root', default=None,
                        help='Validation Dataset root directory path')
    parser.add_argument('--model', default='efficientnet-b0',choices=['mobilenetv2','efficientnet-b0', 'efficientnet-b1'],
                        help='Detector model name')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--resume', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--num_workers', default=12, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--epochs', default=50, type=int,
                        help='Number of epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                        help='Initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optimizer')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--step_size', default=70, type=int,
                        help='Step size for step lr scheduler')
    parser.add_argument('--milestones', default=[20], type=int, nargs='*',
                        help='Milestones for multi step lr scheduler')
    parser.add_argument('--scheduler', default='multi_step',
                        choices=['plateau','step', 'multi_step','cosine'],
                        type=str.lower, help='Use Scheduler')
    parser.add_argument('--optimizer', default='sgd',
                        choices=['adam', 'sgd', 'adamw', 'sgdp'],
                        type=str.lower, help='Use Optimizer')
    parser.add_argument('--loss', default='open',
                        choices=['open', 'ce', 'mse'],
                        type=str.lower, help='Use Optimizer')
    parser.add_argument('--input_size', default=[192,192], type=int,nargs=2,
                        help='input size(width, height)')
    parser.add_argument('--no_background', default=False, action='store_true',
                        help='Use background dataset')

    args = parser.parse_args()
    return args    


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


def save_model(model, epoch, loss, optimizer, path='./checkpoints', postfix="mobilenetV2", cuda=False):
    if postfix == "best":
        target = os.path.join(path, postfix +'.pth')
    else:
        target = os.path.join(path, postfix + "_"+ str(epoch) +'.pth')

    if not os.path.isdir(path):
        os.makedirs(path)
    if cuda:
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, target)
    else:
         torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, target) 

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

    t = [transforms.Resize((args.input_size[1], args.input_size[0])),
        transforms.Grayscale(1),
        #rgbtor(),
        transforms.ToTensor()]
    t = transforms.Compose(t)
    train_dataset = CustomDataset(data_set_path=args.train_dataset_root, transforms=t)
    test_dataset = CustomDataset(data_set_path=args.validation_dataset_root, transforms=t)
    class_names = os.walk(args.train_dataset_root).__next__()[1]
    class_names.sort()
    if "background" in class_names:
        class_names.remove("background")
        class_names.append("background")

    if args.no_background is False:
        if not "background" in class_names:
            raise Exception("There is no background dataset")

    print('number of training data : ',len(train_dataset))
    print('number of test data : ',len(test_dataset))


    # data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size = args.batch_size, drop_last=True, shuffle = True, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size = args.batch_size, drop_last=True, shuffle = True, num_workers=args.num_workers)

    if args.no_background is False:
        if args.model == 'mobilenetv2':
            model = mobilenet_v2(custom_class_num = len(class_names) - 1)
        elif args.model == 'efficientnet-b0':
            model = efficientnet_b0(custom_class_num = len(class_names) - 1)
        elif args.model == 'efficientnet-b1':
            model = efficientnet_b1(custom_class_num = len(class_names) - 1)
    else:
        if args.model == 'mobilenetv2':
            model = mobilenet_v2(custom_class_num = len(class_names))
        elif args.model == 'efficientnet-b0':
            model = efficientnet_b0(custom_class_num = len(class_names))
        elif args.model == 'efficientnet-b1':
            model = efficientnet_b1(custom_class_num = len(class_names))

    if torch.cuda.is_available():
        model = model.to(device)
        model = torch.nn.DataParallel(model)

    #optimizer
    if args.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr = args.lr)
    elif args.optimizer == "sgd":
        optimizer = SGD(model.parameters(),
                        lr=args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = AdamW(model.parameters(), lr = args.lr)
    elif args.optimizer == "sgdp":
        optimizer = SGDP(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    else:
        raise Exception("unknown optimizer")

    #scheduler
    if args.scheduler == "step":
        scheduler = StepLR(optimizer,
                        args.step_size,
                        args.gamma)

    elif args.scheduler == "multi_step":
        scheduler = MultiStepLR(optimizer,
                            args.milestones,
                            args.gamma)

    elif args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer,mode='min',patience=5)

    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, 10, last_epoch =-1)
        
    else:
        scheduler = None

    if args.resume:
        epoch, loss= load_model(model = model, source=args.resume, optimizer=optimizer)
        print("\nresume at epoch: {}, loss: {}\n".format(epoch, loss))
        epoch = epoch + 1
    else:
        epoch = 0
    #open set loss
    if args.no_background is False and args.loss == 'open':
        print("Use open-set Loss")
        criterion = EntropicOpenSetLoss(class_names)
    elif args.loss == 'ce':
        print("Use CE Loss")
        criterion = nn.CrossEntropyLoss()
    else:
        print("Use MSE Loss")
        criterion = nn.MSELoss()    

    # 모델 학습 & 추론
    best_score = 0
    model.train()
    cnt = 0
    total_cnt = len(train_dataset)
    for epoch in range(epoch, args.epochs):
        for data, target in train_loader:
            # data /= 255.
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            if args.loss == 'mse':
                #one-hot encoding
                target = F.one_hot(target,len(class_names))
                loss = criterion(output.to(torch.float32), target.to(torch.float32))
            else:
                loss = criterion(output, target)
            if torch.isfinite(loss): #prevent to go nan
                loss.backward()
                optimizer.step()
            cnt +=  args.batch_size
            print("[%d / %d] - loss=%.3f" % (cnt, total_cnt, loss), end='\r')
        print("Train epoch : {}     Loss : {:3f}".format(epoch, loss.item()))
        save_model(model=model, epoch=epoch, loss=loss, optimizer=optimizer, postfix= args.model, cuda=is_cuda)
        cnt = 0
        #scheduler
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(loss)
            else:
                scheduler.step()
        #모델 평가
        if epoch != 0 and epoch % 5 == 0:
            model.eval()    # 평가시에는 dropout이 OFF 된다.
            correct = 0
            background_correct_num = 0
            wrong = 0
            background_num = 0
            total = 0
            with torch.no_grad():
                for data, target in test_loader:
                    if torch.cuda.is_available():
                        data = data.to(device)
                        target = target.to(device)
                    output = model(data)
                    prediction = torch.max(output[0])
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
                                else:
                                    wrong += 1
                            total += 1
                        else:
                            correct += torch.argmax(output[i]).eq(target[i])
                            total += 1

            if args.no_background is False:
                print("total_wrong: ", wrong)
                print("total_correct: ", correct)
                print("Background num: ",background_num)
                print("Background correct num: ",background_correct_num)
                print('Test set Accuracy without background : {:.2f}%'.format(100. * (correct-background_correct_num) / (total-background_num)))
            print('Test set Accuracy : {:.2f}%'.format(100. * correct / total))
            score = 100. * correct / total
            #save best weight
            if best_score < score:
                best_score = score
                save_model(model=model, epoch=epoch, loss=loss, optimizer=optimizer, postfix="best", cuda=is_cuda)
            model.train()
        
    print("Done")