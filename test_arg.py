import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TEST Configuration')    
    parser.add_argument('--train', action='store_true', help='train mode')
    parser.add_argument('--eval', action='store_true', help='eval mode')
    parser.add_argument('--cuda', action='store_true', help='use gpu')
    parser.add_argument('--ckpt', default=None, type=str, help='e.g., "ckpt/model_{}.pth.tar"')
    args = parser.parse_args()
    print(args)
    print(args.ckpt)
    if not args.ckpt:
        print('hi')
