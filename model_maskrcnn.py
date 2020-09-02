import numpy as np
import torch
import cv2
from dataloader_maskrcnn import DataProcessor
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from skimage import io
from torchvision import transforms
from skimage.color import rgb2gray
import torchvision
from utility.default_transforms import RandomRotate, HorizontalFlip,\
     VerticalFlip, RandomNoise, ToTensor, make_binary
import matplotlib.pyplot as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import warnings
warnings.filterwarnings("ignore")

class MaskRCNN:
    def __init__(self, path_to_dict=None, num_classes=2):
        self.path_to_dict = path_to_dict
        self.model = self._get_model_instance_segmentation(num_classes)
        # Check for CUDA
        train_on_gpu = torch.cuda.is_available()
        if not train_on_gpu:
            self.device = torch.device("cpu")
            print("="*30)
            print("Running on CPU")
            print("=" * 30)
        else:
            print("=" * 30)
            self.device = torch.device("cuda:0")
            print("CUDA is available!")
            print("=" * 30)
        if self.path_to_dict:
            print("Model Weights Loaded")
            print("=" * 30)
            weights = torch.load(path_to_dict, map_location=self.device)
            self.model.load_state_dict(weights)
        # Load model on CUDA/CPU
        self.model.to(self.device)

    # Instantiate torchvision MaskRCNN model
    def _get_model_instance_segmentation(self, num_classes):
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        # get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           num_classes)
        return model

    # Compose transformations
    def _get_default_transforms(self):
        custom_transforms = []
        custom_transforms.extend([RandomRotate(), HorizontalFlip(),
                                  VerticalFlip(), RandomNoise()])
        return transforms.Compose(custom_transforms)

    # Computer bounding box iou
    def _bb_intersection_over_union(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = abs(max(0, xB - xA + 1) * max(0, yB - yA + 1))
        if interArea == 0:
            return 0.0
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # delete variable to save memory
        del xA, yA, xB, yB, interArea, boxAArea, boxBArea
        return iou

    # Computer mask iou
    def _get_iou_vector(self, target, prediction):
        intersection = np.logical_and(target, prediction)
        union = np.logical_or(target, prediction)
        run_iou = np.sum(intersection) / np.sum(union)
        #delete variable to save memory
        del intersection, union
        return run_iou

    # Calculate prediction masks and computer iou
    def _calculate_maskiou(self, gt_mask, pred_mask, shape=256, thresh=0.5):
        ground_mask = np.zeros((shape, shape))
        predicted_mask = np.zeros((shape, shape))
        if pred_mask.shape[0] == gt_mask.shape[0]:
            if pred_mask.shape[0] > 1 and gt_mask.shape[0] > 1:
                for i in range(pred_mask.shape[0]):
                    ground_mask += gt_mask[i]
                    predicted_mask += pred_mask[i]
            else:
                ground_mask = gt_mask[0]
                predicted_mask = pred_mask[0]
            # Set threshold for masks
            predicted_mask = predicted_mask > thresh
            predicted_mask = (predicted_mask * 255)
            predicted_mask = make_binary(predicted_mask.astype(np.uint8))
            ground_mask = make_binary(ground_mask)
            # Calculate iou
            mask_iou = self._get_iou_vector(ground_mask, predicted_mask)
            # Delete variable to save memory
            del ground_mask, predicted_mask
            return mask_iou
        else:
            print("Shape mismatch of ground-truth and prediction mask")
            return 0.0

    # Calculate bounding box and mask iou
    def _calculate_ious(self, annotations, output):
        bbox_iou_over_batch = 0.0
        mask_iou_over_batch = 0.0
        for i in range(len(output)):
            out_truth_bbox = annotations[i]['boxes'].numpy().squeeze()
            out_truth_mask = annotations[i]['masks'].numpy().squeeze()
            out_prediction_bbox = output[i]['boxes']
            out_prediction_mask = output[i]['masks']
            if len(out_prediction_bbox) == 0 or len(out_prediction_bbox) == 0:
                bbox_iou = 0.0
                mask_iou = 0.0
            else:
                bbox_iou = 0.0
                for index, arg in enumerate(out_truth_bbox):
                    bbox_iou += self._bb_intersection_over_union(arg, out_prediction_bbox[index])
                mask_iou = self._calculate_maskiou(out_truth_mask, out_prediction_mask)
            bbox_iou_over_batch += bbox_iou
            mask_iou_over_batch += mask_iou
        bbox_batch_iou = bbox_iou_over_batch / len(output)
        mask_batch_iou = mask_iou_over_batch / len(output)
        # Delete to free from memory
        del out_truth_bbox, out_truth_mask, out_prediction_bbox, \
            out_prediction_mask, bbox_iou_over_batch, mask_iou_over_batch
        return bbox_batch_iou, mask_batch_iou

    # Keep two boxes with high scores
    def _keep_Highscore(self, output):
        processed_output = []
        for i in range(len(output)):
            processed_dict = {}
            out = output[i]['boxes'].cpu().numpy().squeeze()
            masks = output[i]['masks'].cpu().numpy().squeeze()
            if len(out) == 0:
                processed_dict['boxes'] = []
                processed_dict['masks'] = []
            elif len(out) > 2:
                scores = output[i]['scores'].cpu().squeeze()
                top_score, top_index = scores.topk(2)
                new_boxes = []
                new_masks = []
                for arg in top_index:
                    new_boxes.append(out[arg])
                    new_masks.append(masks[arg])
                new_boxes = np.array(new_boxes)
                new_masks = np.array(new_masks)
                processed_dict['boxes'] = new_boxes
                processed_dict['masks'] = new_masks
            else:
                processed_dict['boxes'] = out
                processed_dict['masks'] = masks
            processed_output.append(processed_dict)
        # Delete variable to free memory
        try:
            del processed_dict, out, masks, scores, new_masks, new_boxes, top_score, top_index
        except Exception as e:
            pass
        return processed_output

    def _collate_fn(self, batch):
        return tuple(zip(*batch))

    # Train function
    def train_model(self, path_to_images=None, path_to_masks=None, transformation=None, val_percent=0.2, batch_size=5, lr_rate=1e-3, num_epochs=100):
        if path_to_images and path_to_masks: # lr = 0.005
            if transformation is None:
                dataset = DataProcessor(imgs_dir=path_to_images, masks_dir=path_to_masks, transformations=self._get_default_transforms())
            else:
                dataset = DataProcessor(imgs_dir=path_to_images, masks_dir=path_to_masks, transformations=transformation)
            # Split into training and validation
            n_val = int(len(dataset) * val_percent)
            n_train = len(dataset) - n_val
            train, val = random_split(dataset, [n_train, n_val])
            print("Images for Training:", n_train)
            print("Images for Validation:", n_val)
            trainloader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=self._collate_fn)
            validloader = DataLoader(val, batch_size=batch_size, shuffle=True, collate_fn=self._collate_fn)
            print("=" * 30)

            # Instantiate model parameters
            params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = optim.SGD(params, lr=lr_rate, momentum=0.9, weight_decay=1e-6) #0.0005
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
            global_bbox_ious, global_mask_ious, global_train_loss = [], [], []
            best_wghts_found = False
            mask_iou_min = 0.0
            loss_metrics_avg = {"loss_classifier": [], "loss_box_reg": [], "loss_mask": [], "loss_rpn_box_reg": []}

            # Training and validatin loop
            for epoch in range(num_epochs):
                train_loss, bbox_iou, mask_iou = 0.0, 0.0, 0.0
                loss_metrics = {"loss_classifier": 0.0, "loss_box_reg": 0.0, "loss_mask": 0.0, "loss_rpn_box_reg": 0.0}
                epoch_loss = []
                for imgs, annotations in trainloader:
                    imgs = list(img.to(self.device, dtype=torch.float) for img in imgs)
                    annotations = [{k: v.to(self.device) for k, v in t.items()} for t in annotations]
                    optimizer.zero_grad()
                    loss_dict = self.model(imgs, annotations)
                    # Store losses
                    for arg in loss_dict:
                        if arg in loss_metrics:
                            loss_metrics[arg] += float(loss_dict[arg].item()) * len(imgs)

                    # Aggregate loss and backpropagate
                    losses = sum(loss for loss in loss_dict.values())
                    losses.backward()
                    optimizer.step()
                    epoch_loss.append(float(losses.item() * len(imgs)))
                    detached_loss = losses.detach().item() * len(imgs)
                    train_loss += detached_loss

                # Validation
                scheduler.step(np.mean(epoch_loss))
                with torch.no_grad():
                    self.model.eval()
                    for images, annotations in validloader:
                        imgs = list(img.to(self.device, dtype=torch.float) for img in images)
                        annotations = [{k: v for k, v in t.items()} for t in annotations]
                        output = self.model(imgs)
                        # Process output from model
                        output_numpy = self._keep_Highscore(output)
                        bbox_iou_batch, mask_iou_batch = self._calculate_ious(annotations, output_numpy)
                        bbox_iou += bbox_iou_batch
                        mask_iou += mask_iou_batch

                self.model.train()
                train_loss = train_loss / len(trainloader)
                avg_bbox_iou = bbox_iou / len(validloader)
                avg_mask_iou = mask_iou / len(validloader)
                for arg in loss_metrics:
                    if arg in loss_metrics_avg:
                        loss_metrics_avg[arg].append(loss_metrics[arg]/len(trainloader))

                # Append loss and iou's
                global_train_loss.append(train_loss)
                global_bbox_ious.append(avg_bbox_iou)
                global_mask_ious.append(avg_mask_iou)
                # Save model
                print("Epoch:{}/{}\t Training Loss:{:.6f}\t Average BBox IoU: {:.6f} \t Average Mask IoU: {:.6f}".format(epoch+1, num_epochs,
                                                                                                                         train_loss, avg_bbox_iou,
                                                                                                                         avg_mask_iou))
                if avg_mask_iou > mask_iou_min:
                    print("Average mask IoU increased: ({:.6f} --> {:.6f}).  Saving model ...".format(mask_iou_min, avg_mask_iou))
                    print("-" * 40)
                    # Save model
                    torch.save(self.model.state_dict(), 'checkpoints/LungMaskRCNN.pth')
                    mask_iou_min = avg_mask_iou
                    best_wghts_found = True
                else:
                    # Load previous best weights if no improvements
                    if best_wghts_found:
                        weights = torch.load("checkpoints/LungMaskRCNN.pth")
                        self.model.load_state_dict(weights)
                        del weights
                # Remove variables from memory
                del train_loss, bbox_iou, mask_iou, loss_metrics, avg_bbox_iou, avg_mask_iou, epoch_loss, \
                    output_numpy, bbox_iou_batch, mask_iou_batch,

            # Save loss and iou plots
            plt.plot(loss_metrics_avg["loss_classifier"], label='Classifier loss')
            plt.plot(loss_metrics_avg["loss_box_reg"], label='Box-reg loss')
            plt.plot(loss_metrics_avg["loss_mask"], label='Mask loss')
            plt.plot(loss_metrics_avg["loss_rpn_box_reg"], label='RPN-box loss')
            plt.plot(global_train_loss, label='Total loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(frameon=False)
            plt.savefig('mask_rcnn_train_loss.png')
            plt.clf()

            plt.plot(global_bbox_ious, label='Bounding Box IoU')
            plt.plot(global_mask_ious, label='Mask IoU')
            plt.xlabel('Epoch')
            plt.ylabel('IoU')
            plt.legend(frameon=False)
            plt.savefig('mask_rcnn_ious.png')
            plt.clf()
        else:
            print("Path to images and masks required!")

    def runtestset(self, path_to_images=None, path_to_masks=None, pass_one_batch=False, batch_size=5):
        if path_to_images and path_to_masks:
            dataset = DataProcessor(imgs_dir=path_to_images, masks_dir=path_to_masks, transformations=None)
            testloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=self._collate_fn)
            # Evaluate model on test set
            bbox_iou, mask_iou = 0.0, 0.0
            with torch.no_grad():
                self.model.eval()
                if pass_one_batch:
                    images, annotations = iter(testloader).next()
                    imgs = list(img.to(self.device, dtype=torch.float) for img in images)
                    output = self.model(imgs)
                    output_numpy = self._keep_Highscore(output)
                    # Show prediction results
                    self._show_predictions(imgs, annotations, output_numpy)
                else:
                    for images, annotations in testloader:
                        imgs = list(img.to(self.device, dtype=torch.float) for img in images)
                        annotations = [{k: v for k, v in t.items()} for t in annotations]
                        output = self.model(imgs)
                        output_numpy = self._keep_Highscore(output)
                        bbox_iou_batch, mask_iou_batch = self._calculate_ious(annotations, output_numpy)
                        bbox_iou += bbox_iou_batch
                        mask_iou += mask_iou_batch
                    avg_bbox_iou = bbox_iou / len(testloader)
                    avg_mask_iou = mask_iou / len(testloader)
                    print("Average BBOX IoU:", avg_bbox_iou)
                    print("Average Mask IoU:", avg_mask_iou)

    def _show_predictions(self, images, gt_annotations, output, mask_shape=256):
        for i in range(len(output)):
            image = images[i][0].cpu().numpy()
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            # GT
            gt_boxes = gt_annotations[i]['boxes'].numpy()
            gt_masks = gt_annotations[i]['masks'].numpy()
            # Predictions
            bbox = output[i]['boxes']
            masks = output[i]['masks']
            cloned_img = img_rgb.copy()
            # Draw Predictoins
            predicted_mask = np.zeros((mask_shape, mask_shape))
            GT_mask = np.zeros((mask_shape, mask_shape))
            for j in range(bbox.shape[0]):
                predicted_mask += masks[j]
                GT_mask += gt_masks[j]
                cv2.rectangle(img_rgb, (bbox[j][0], bbox[j][1]), (bbox[j][2], bbox[j][3]), color=(0, 0, 255), thickness=2)
                cv2.rectangle(cloned_img, (gt_boxes[j][0], gt_boxes[j][1]), (gt_boxes[j][2], gt_boxes[j][3]), color=(0, 0, 255),
                              thickness=2)
            # Disply results
            predicted_mask = predicted_mask > 0.5
            predicted_mask = (predicted_mask * 255)
            predicted_mask = make_binary(predicted_mask.astype(np.uint8))

            # Display GT
            plt.imshow(cloned_img)
            plt.show()
            plt.imshow(GT_mask)
            plt.show()

            # Disply Predictions
            plt.imshow(img_rgb)
            plt.show()
            plt.imshow(predicted_mask)
            plt.show()

    # Inference function
    def get_prediction(self, path_to_image=None):
        if path_to_image:
            data = []
            image = rgb2gray(io.imread(path_to_image))
            dataset = DataProcessor(imgs_dir=None, masks_dir=None)
            processed = dataset.preprocess(image, expand_dim=False, adjust_label=False, normalize=True)
            data.append(processed.to(self.device, dtype=torch.float))
            with torch.no_grad():
                self.model.eval()
                output = self.model(data)
            img = image.astype('float32')
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            bbox = np.squeeze(output[0]['boxes'].cpu().numpy())
            return bbox

