from .libs import *
from ._utils import bbox_overlaps

def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor, device, is_training=True) -> torch.Tensor:
    y_true = y_true.to(device).flatten()
    y_pred = y_pred.to(device).flatten()

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1

def eval_model(model, data_loader, device):
    model.eval()
    map50 = []
    f1score = []
    for images, targets in tqdm(data_loader):
        images = [image.to(device) for image in images]
        targets = [{k: v.cpu() for k, v in t.items()} for t in targets]

        with torch.no_grad():
            predictions = model(images) # Return (anchor=100, ..., ..., ...)

        for j in range(len(predictions)):
            idx_text = (targets[j]["labels"] == 1)
            target_box = targets[j]["boxes"][idx_text]
            target_mask = targets[j]["masks"][idx_text]

            idx_text = (predictions[j]["labels"] == 1)
            pred_box = predictions[j]["boxes"][idx_text]
            pred_mask = predictions[j]["masks"][idx_text]

            f1score_temp = []
            for i in range(len(pred_mask)):
                f1score_temp.append(f1_loss(target_mask, pred_mask[i], device))
            try:
                f1score.append(max(f1score_temp).detach().cpu())
            except:
                f1score.append(0)

            ious_score = bbox_overlaps(pred_box.cpu(), target_box).numpy()
            map = np.mean(ious_score[range(len(ious_score)), np.argmax(ious_score, -1)] >= 0.5)
            map50.append(map)
    return np.mean(map50), np.mean(f1score)