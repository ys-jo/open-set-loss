import os
import torch
from torchvision import  transforms
import argparse
from dataset import CustomDataset
from model import mobilenet_v2
from efficientnet import efficientnet_b0, efficientnet_b1
from accuracy import Accuracy
import numpy as np
from PIL import Image
import torchvision

def parser():
    parser = argparse.ArgumentParser(description='evaluate network')
    parser.add_argument('--model', default='efficientnet-b0',choices=['mobilenetv2','efficientnet-b0', 'efficientnet-b1'],
                        help='Detector model name')
    parser.add_argument('--dataset_root', default=None, type=str,
                        help='Dataset root directory path')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_workers', default=12, type=int,
                        help='Number of workers used in dataloading')  
    # model parameter
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
    parser.add_argument('--onnx', default=None, type=str,
                        help='Use onnxruntime')
    parser.add_argument('--dx_sim', default=None, type=str,
                        help='Use dx_sim')
    parser.add_argument('--json', default=None, type=str,
                        help='Use dx_sim')
    args = parser.parse_args()
    return args


class ToNPUTensor:
    def __call__(self, img):
        img = np.array(img).astype(np.float32)
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=0)

        # transposed_img = np.transpose(img, (2, 0, 1))
        return img
    
def create_simulator_input(img, preprocessor):
    """Create input for simulator with using image file"""
    pil_image = img.convert('RGB') 
    open_cv_image = np.array(pil_image) 
    # Convert RGB to BGR 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    preprocessed = preprocessor(open_cv_image)
    return preprocessed.astype(np.uint8)

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


def export_classificaton_model(model_name, size, model):
    w = size[0]
    h = size[1]

    x = torch.rand(1, 1, h, w)

    if torch.cuda.is_available():
        x = x.cuda()
    filename = model_name + '.onnx'
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
def collate(batch):
    imgs = []
    targets = []

    for (img, target) in batch:
        imgs.append(img)
        targets.append(target)

    return imgs[0], torch.tensor(targets)


if __name__ == "__main__":
    args = parser()
    #use cuda
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')

    # dataset_root check
    if not os.path.isdir(args.dataset_root):
        raise Exception("There is no dataset_root dir") 
    if args.dx_sim:
        t = [transforms.Resize((args.input_size[1], args.input_size[0])),
            transforms.Grayscale(1),
            ToNPUTensor()]
    
        t = transforms.Compose(t)
        test_dataset = CustomDataset(data_set_path=args.dataset_root, transforms=t)
    else:
        t = [transforms.Resize((args.input_size[1], args.input_size[0])),
            transforms.Grayscale(1),
            transforms.ToTensor()]
    
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
    if args.onnx:
        args.batch_size = 1
        import onnxruntime
        if torch.cuda.is_available():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        session = onnxruntime.InferenceSession(args.onnx, providers=providers)
        output_names = [x.name for x in session.get_outputs()]
        meta = session.get_modelmeta().custom_metadata_map  # metadata
        if "stride" in meta:
            stride, names = int(meta["stride"]), eval(meta["names"])
    elif args.dx_sim:
        from dx_simulator import Simulator
        args.batch_size = 1
        simulator = Simulator(
            opt_model_path=os.path.join(args.dx_sim, "opt.model"),
            pre_model_path=os.path.join(args.dx_sim, "pre.model"),
            cpu_model_path=os.path.join(args.dx_sim, "cpu.model"),
            sequence_path=os.path.join(args.dx_sim, "simulator_info.pb"),
            config_path=args.json,
            mode="inference",
        )
        preprocessor = simulator.get_preprocessing()
    else:
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

        _=load_model(model, source=args.weight, eval=1)
        if args.export:
            if torch.cuda.is_available():
                model = model.cuda()
            export_classificaton_model(args.model, args.input_size, model)
        if torch.cuda.is_available():
            model = model.cuda()
            model = torch.nn.DataParallel(model)
        model.eval()    # 평가시에는 dropout이 OFF 된다.

    print('number of test data : ', len(test_dataset))
    # data loader
    if args.dx_sim:
        test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate)
    else:
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size = args.batch_size, drop_last=True, shuffle = True, num_workers = args.num_workers)
   

    metric = Accuracy(topk=args.topk, remove = len(class_names)-1, no_background = args.no_background)
    metric.reset()
    topk = args.topk
    remainder = len(test_dataset)%args.batch_size

    #모델 평가
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

        if args.onnx:
            imgs = data.cpu().numpy()  # torch to numpy
            outputs = session.run(output_names, {session.get_inputs()[0].name: imgs})
            output = torch.tensor(outputs[0])
            target = target.to('cpu')
        elif args.dx_sim:
            imgs = data.astype(np.uint8)
            #  imgs = create_simulator_input(data, preprocessor)
            outputs = simulator.run(output_names = [simulator.get_outputs()[0].name],  data = {simulator.get_inputs()[0].name : imgs})
            output = torch.FloatTensor(outputs[0])
            target = target.to('cpu')
        else:
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
        metric.match(output, target)
    accuracy = metric.get_result()
    # print results
    print("==================================================")
    print("========배경클래스 없이 정확도 측정===============")
    print("accuracy (without background class) = %.3f %%" % accuracy)
    for i in range(len(class_names)-1):
        print('Accuracy of %s: %.3f %%' % (class_names[i], 100 * class_correct[i] / class_total[i]))

    print("==================================================")
    print("========배경클래스 포함 정확도 측정===============")
    if args.no_background is False:
        print("total_wrong: ", wrong)
        print("total_correct: ", int(correct.detach().cpu()))
        print("Background num: ", background_num)
        print("Background correct num: ",background_correct_num)
        print('Test set Accuracy without background : {:.2f}%'.format(100. * (correct-background_correct_num) / (total-background_num)))
    print('Test set Accuracy : {:.2f}%'.format(100. * correct / total))
