import argparse


def get_params():
    parser = argparse.ArgumentParser(description='config parameters for memAE_MNIST.py')
    
    parser.add_argument('--epoch', type=int, default=25, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='training mini batch size')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold for entropy')
    parser.add_argument('--sampling_rate', type=float, default=0.05, help='sampling ratio')
    parser.add_argument('--arch', type=str, default="resnet18", help='backbone architecture')
    parser.add_argument('--data_path', type=str, default="./data", help='data directory')
    parser.add_argument('--device', type=str, default="cuda:0", help='training device')
    parser.add_argument('--ckpt_name', type=str, default="model.pth", help='checkpoint name')

    return parser