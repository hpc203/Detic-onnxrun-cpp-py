import argparse
import cv2
import numpy as np
import onnxruntime as ort


class Detic():
    def __init__(self, modelpath, detection_width=800, confThreshold=0.8):
        # net = cv2.dnn.readNet(modelpath)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(modelpath, so)
        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.max_size = detection_width
        self.confThreshold = confThreshold
        self.class_names = list(map(lambda x: x.strip(), open('imagenet_21k_class_names.txt').readlines()))
        self.assigned_colors = np.random.randint(0,high=256, size=(len(self.class_names), 3)).tolist()

    def preprocess(self, srcimg):
        im_h, im_w, _ = srcimg.shape
        dstimg = cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB)
        if im_h < im_w:
            scale = self.max_size / im_h
            oh, ow = self.max_size, scale * im_w
        else:
            scale = self.max_size / im_w
            oh, ow = scale * im_h, self.max_size

        max_hw = max(oh, ow)
        if max_hw > self.max_size:
            scale = self.max_size / max_hw
            oh *= scale
            ow *= scale
        ow = int(ow + 0.5)
        oh = int(oh + 0.5)
        dstimg = cv2.resize(dstimg, (ow, oh))
        return dstimg

    def post_processing(self, pred_boxes, scores, pred_classes, pred_masks, im_hw, pred_hw):
        scale_x, scale_y = (im_hw[1] / pred_hw[1], im_hw[0] / pred_hw[0])

        pred_boxes[:, 0::2] *= scale_x
        pred_boxes[:, 1::2] *= scale_y
        pred_boxes[:, [0, 2]] = np.clip(pred_boxes[:, [0, 2]], 0, im_hw[1])
        pred_boxes[:, [1, 3]] = np.clip(pred_boxes[:, [1, 3]], 0, im_hw[0])

        threshold = 0
        widths = pred_boxes[:, 2] - pred_boxes[:, 0]
        heights = pred_boxes[:, 3] - pred_boxes[:, 1]
        keep = (widths > threshold) & (heights > threshold)

        pred_boxes = pred_boxes[keep]
        scores = scores[keep]
        pred_classes = pred_classes[keep]
        pred_masks = pred_masks[keep]

        # mask_threshold = 0.5
        # pred_masks = paste_masks_in_image(
        #     pred_masks[:, 0, :, :], pred_boxes,
        #     (im_hw[0], im_hw[1]), mask_threshold
        # )

        pred = {
            'pred_boxes': pred_boxes,
            'scores': scores,
            'pred_classes': pred_classes,
            'pred_masks': pred_masks,
        }
        return pred

    def draw_predictions(self, img, predictions):
        height, width = img.shape[:2]
        default_font_size = int(max(np.sqrt(height * width) // 90, 10))
        boxes = predictions["pred_boxes"].astype(np.int64)
        scores = predictions["scores"]
        classes_id = predictions["pred_classes"].tolist()
        # masks = predictions["pred_masks"].astype(np.uint8)
        num_instances = len(boxes)
        print('detect', num_instances, 'instances')
        for i in range(num_instances):
            x0, y0, x1, y1 = boxes[i]
            color = self.assigned_colors[classes_id[i]]
            cv2.rectangle(img, (x0, y0), (x1, y1), color=color,thickness=default_font_size // 4)
            text = "{} {:.0f}%".format(self.class_names[classes_id[i]], round(scores[i],2) * 100)
            cv2.putText(img, text, (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=1, lineType=cv2.LINE_AA)
        return img

    def detect(self, srcimg):
        im_h, im_w = srcimg.shape[:2]
        dstimg = self.preprocess(srcimg)
        pred_hw = dstimg.shape[:2]
        input_image = np.expand_dims(dstimg.transpose(2, 0, 1), axis=0).astype(np.float32)

        # Inference
        pred_boxes, scores, pred_classes, pred_masks = self.session.run(None, {self.input_name: input_image})
        preds = self.post_processing(pred_boxes, scores, pred_classes, pred_masks, (im_h, im_w), pred_hw)
        return preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgpath", type=str, default='desk.jpg', help="image path")
    parser.add_argument("--confThreshold", default=0.5, type=float, help='class confidence')
    parser.add_argument("--modelpath", type=str, default='weights/Detic_C2_R50_640_4x_in21k.onnx', help="onnxmodel path")
    args = parser.parse_args()

    mynet = Detic(args.modelpath, confThreshold=args.confThreshold)
    srcimg = cv2.imread(args.imgpath)
    preds = mynet.detect(srcimg)
    srcimg = mynet.draw_predictions(srcimg, preds)

    # cv2.imwrite('result.jpg', srcimg)
    winName = 'Deep learning Detic in ONNXRuntime'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.imshow(winName, srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
