from networks import *

class MaskRCNN(object):
    def __init__(self, weights="checkpoints/latest_model.pth", backbone='resnet101'):
        self.weights = weights
        self.model = models_inference[backbone]
        self.model_detect = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device=self.device)
        self.model.load_state_dict(torch.load(self.weights, map_location=self.device)['state_dict'])
        self.model.eval()

        self.label_to_num = {'hair': 1}
        self.class_names = ['BG', 'hair']

    def check_type(self, img_path):
        if type(img_path) == str:
            if img_path.endswith(('.jpg', '.png', '.jpeg')):
                img = cv2.imread(img_path)
            else:
                raise Exception("Please input a image file")
        elif type(img_path) == np.ndarray:
            img = img_path
        return img

    def crop(self, img):
        results = self.model_detect(img)
        t = results.pandas().xyxy[0]  # or .show(), .save(), .crop(), .pandas(), etc.
        bbox = np.int32(np.array(t)[:, :4][np.where(np.array(t)[:, 6] == 'person')])
        if len(bbox) > 1:
            area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
            bbox = bbox[np.where(max(area))]
        # Frame ma khong detect duoc thi lay ca frame luon
        elif len(bbox) < 1:
            h, w, _ = img.shape
            bbox = np.array([[0, 0, w, h]], dtype=np.uint32)

        x1, y1, x2, y2 = bbox[0]
        return x1, y1, x2, y2, img[y1:y2, x1:x2]

    def visualize(self,img, label, color = (0, 255, 0)):
        if color:
            label[:,:,0][np.where(label[:,:,0]==255)] = color[0]
            label[:,:,1][np.where(label[:,:,1]==255)] = color[1]
            label[:,:,2][np.where(label[:,:,2]==255)] = color[2]
        return cv2.addWeighted(img, 0.6, label, 0.4, 0)

    def predict(self, img_path, is_visualize=True):
        img = self.check_type(img_path)

        img_predict = torch.as_tensor(img[:, :, ::-1].transpose(2, 0, 1) / 255.0, dtype=torch.float32).cpu()

        with torch.no_grad():
            prediction = self.model(img_predict[None, :])[0]

        labels = prediction["labels"]  # tensor([1])
        scores = prediction["scores"]  # tensor([0.99])
        boxes = np.int32(prediction['boxes'].cpu().numpy())  # tensor([[  x_min,   y_min,  x_max,   y_max]])

        masks = prediction["masks"].squeeze(1)
        masks = np.round(masks.cpu().numpy())
        final_mask = np.zeros((img.shape[0], img.shape[1]))
        for j in range(len(masks)):
            final_mask += masks[j]
        masks = np.clip(final_mask, 0, 255)

        masks = np.uint8(cv2.merge([final_mask, final_mask, final_mask]) * 255)

        if is_visualize:
            return self.visualize(img, masks)

        return cv2.addWeighted(img, 0.85, masks, 0.4, 0)

def webcam():
    print("Using webcam, press [q] to exit, press [s] to save")
    cap = cv2.VideoCapture(0)
    cap.set(3, 1920)
    cap.set(4, 1080)
    with alive_bar(theme='musical', length=200) as bar:
        while True:
            _, frame = cap.read()
            frame = cv2.flip(frame, 1)
            start = time.time()
            frame = MaskRCNNPredictor.predict(frame)
            fps = round(1 / (time.time() - start), 2)
            cv2.putText(frame, "FPS : " + str(fps), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
            cv2.imshow('Prediction', frame + 30)
            k = cv2.waitKey(20) & 0xFF
            if k == ord('s'):
                os.makedirs('results/', exist_ok=True)
                cv2.imwrite('results/' + str(time.time()) + '.jpg', frame)
            if k == ord('q'):
                break
            bar()

def image(path_img):
    start = time.time()
    img = MaskRCNNPredictor.predict(path_img)
    print('Time inference:', round((time.time() - start) * 1e3, 2), 'ms')
    cv2.imshow('MaskRCNN predict', img)
    cv2.waitKey(0)
    # cv2.imwrite('datasets/results/' + path_img.split('/')[-1], img)

if __name__ == '__main__':
    MaskRCNNPredictor = MaskRCNN(weights='checkpoints/best_model_r18_87e.pth', backbone='resnet18')
    image('dataset/Figaro_1k_png/test/images/79.jpg')