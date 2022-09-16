from networks import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="dataset/Figaro_1k_png/", help="Path to input image")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--pretrained", type=str, default='checkpoints/')
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=2)
    args = parser.parse_args()
    print("           ⊱ ──────ஓ๑♡๑ஓ ────── ⊰")
    print("🎵 hhey, arguments are here if you need to check 🎵")
    for arg in vars(args):
        print("{:>15}: {:>30}".format(str(arg), str(getattr(args, arg))))
    print()
    return args

def eval():
    args = get_args()
    model = resnet50_maskRCNN(args.num_classes, True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    start_epoch = 0
    # Load pretrained if exist
    if os.path.exists(os.path.join(args.pretrained, 'lastest_model.pth')):
        checkpoint = torch.load(os.path.join(args.pretrained, 'latest_model.pth'), map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        mAP = checkpoint['mAP']
        print('Resume training from ---{}--- have mIoU = {}, start at epoch: {} \n'.format(args.pretrained, mAP, start_epoch))

    dataset_val = HairDataset(path_dataset=args.root, transforms=get_transform(train=False), mode='val')

    val_dataloader = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch,
                                                 shuffle=False, num_workers=args.num_workers,
                                                 collate_fn=collate_fn)

    mm, f1 = eval_model(model, val_dataloader, device)
    print(f"mAP@50: {mm} -- F1Score: {f1}")
    return mm, f1

if __name__ == '__main__':
    mm, f1 = eval()