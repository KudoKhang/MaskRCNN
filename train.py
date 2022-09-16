from networks import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="dataset/Figaro_1k_png/", help="Path to input image")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--pretrained", type=str, default='checkpoints/')
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=2)
    args = parser.parse_args()
    print("           ⊱ ──────ஓ๑♡๑ஓ ────── ⊰")
    print("🎵 hhey, arguments are here if you need to check 🎵")
    for arg in vars(args):
        print("{:>15}: {:>30}".format(str(arg), str(getattr(args, arg))))
    print()
    return args

def train():
    args = get_args()

    # Init model
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
        print('Resume training from ---{}--- have mIoU = {}, start at epoch: {} \n'.format(args.pretrained, mAP,
                                                                                           start_epoch))
    # Dataloader
    dataset_train = HairDataset(path_dataset=args.root, transforms=get_transform(train=True), mode='train')
    train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch,
                                              shuffle=True, num_workers=args.num_workers,
                                              collate_fn=collate_fn)

    dataset_val = HairDataset(path_dataset=args.root, transforms=get_transform(train=False), mode='val')
    val_dataloader = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch,
                                                   shuffle=False, num_workers=args.num_workers,
                                                   collate_fn=collate_fn)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Loss
    scaler = torch.cuda.amp.GradScaler()

    n_batch = len(train_dataloader)
    max_map = 0

    for epoch in range(start_epoch, args.epoch):
        model.train()
        losses_record = []
        with tqdm(train_dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}")
            for batch_idx, (images, targets) in enumerate(tepoch):
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                with torch.cuda.amp.autocast():
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                # Backward and optimize
                optimizer.zero_grad()
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()

                losses_record.append(losses.item())
                tepoch.set_postfix(loss=losses_record[-1])

                # Save weights
                os.makedirs(args.pretrained, exist_ok=True)
                if batch_idx >= n_batch - 1:
                    mm, f1 = eval_model(model, val_dataloader, device)
                    ll = np.mean(losses_record)
                    if max_map < mm:
                        max_map = mm
                        states = {
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'mAP': max_map
                        }
                        torch.save(states, os.path.join(args.pretrained, 'best_model.pth'))
                        tepoch.set_postfix(loss=ll, max_map=mm, f1score=f1, save_weight='True')
                    else:
                        tepoch.set_postfix(loss=ll, max_map=mm, f1score=f1, save_weight='False')
                        states = {
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'mAP': max_map
                        }
                        torch.save(states, os.path.join(args.pretrained, 'latest_model.pth'))
            lr_scheduler.step(ll)

if __name__ == '__main__':
    train()