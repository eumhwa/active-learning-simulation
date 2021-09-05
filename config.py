import argparse


def get_params():
    parser = argparse.ArgumentParser(description='config parameters for memAE_MNIST.py')
    
    parser.add_argument('--n_exp', type=int, default=10, help='size of simulation')
    parser.add_argument('--n_iter', type=int, default=3, help='number of iteration')
    parser.add_argument('--epoch', type=int, default=25, help='training epochs')
    parser.add_argument('--re_epoch', type=int, default=10, help='retraining epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='training mini batch size')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold for entropy')
    parser.add_argument('--sampling_rate', type=float, default=0.1, help='sampling ratio')
    parser.add_argument('--arch', type=str, default="wide_resnet50_2", help='backbone architecture')
    parser.add_argument('--last_class_id', type=int, default=10, help='last class id for defining training set')
    parser.add_argument('--data_path', type=str, default="/content/drive/MyDrive/Colab_dataset/flower_data/", help='data directory')
    parser.add_argument('--device', type=str, default="cuda:0", help='training device')
    parser.add_argument('--ckpt_name', type=str, default="model.pth", help='checkpoint name')

    return parser