from .libs import *

class HairDataset(torch.utils.data.Dataset):
    def __init__(self, path_dataset="dataset/Figaro_1k_png", transforms=None, mode='train', max_size=512):
        self.path_dataset = path_dataset
        self.transforms = transforms
        self.mode = mode
        self.max_size = max_size

        self.DATA_PATH = os.path.join(os.getcwd(), self.path_dataset)
        self.train_path, self.val_path, self.test_path = [os.path.join(self.DATA_PATH, x) for x in
                                                          ['train', 'val', 'test']]

        if self.mode == 'train':
            self.data_files = self.get_files(self.train_path)
            self.label_files = [self.get_label_file(f, 'images', 'masks') for f in self.data_files]
        elif self.mode == 'val':
            self.data_files = self.get_files(self.val_path)
            self.label_files = [self.get_label_file(f, 'images', 'masks') for f in self.data_files]
        elif self.mode == 'test':
            self.data_files = self.get_files(self.test_path)
            self.label_files = [self.get_label_file(f, 'images', 'masks') for f in self.data_files]
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

    def get_files(self, data_folder):
        return glob("{}/*.{}".format(os.path.join(data_folder, 'images'), 'jpg'))

    def get_label_file(self, data_path, data_dir, label_dir):
        data_path = data_path.replace(data_dir, label_dir)
        fname, _ = data_path.split('.')
        return "{}.{}".format(fname, 'png')

    def resize(self, data, label, max_size=self.max_size):
        w, h = data.size
        max = h if h >= w else w
        new_size = (int(max_size * w / h), max_size) if max == h else (max_size, int(max_size * h / w))
        data = data.resize(new_size, Image.ANTIALIAS)
        label = label.resize(new_size, Image.ANTIALIAS)
        return data, label

    def image_loader(self, data_path, label_path):
        data = Image.open(data_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        return self.resize(data, label)
    
    def __getitem__(self, index):
        data_path, label_path = self.data_files[index], self.label_files[index]
        img, label = self.image_loader(data_path, label_path)

        labels = [1]

        # get bounding box coordinates for each mask
        boxes = []
        ymin = min(np.where(np.array(label) == 255)[0])
        ymax = max(np.where(np.array(label) == 255)[0])

        xmin = min(np.where(np.array(label) == 255)[1])
        xmax = max(np.where(np.array(label) == 255)[1])

        boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.as_tensor(np.array(label), dtype=torch.uint8)
        masks = torch.unsqueeze(masks, 2).permute(2, 0, 1) / 255.0
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        # return int(len([name for name in os.listdir(os.path.join(self.DATA_PATH, self.mode, 'images')) if name.endswith('jpg')]) / 20)
        return len([name for name in os.listdir(os.path.join(self.DATA_PATH, self.mode, 'images')) if name.endswith('jpg')])
