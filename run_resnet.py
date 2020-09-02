from sklearn.metrics import precision_score, accuracy_score, recall_score, \
    roc_curve, roc_auc_score, auc
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
from keras.utils import np_utils
import torchvision.transforms as transforms
from model_resnet import ResidualBlock, ResNet
from dataloader_resnet import DataProcessor
import torch.nn.functional as F
from itertools import cycle
import matplotlib.pyplot as plt
from scipy import interp
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix as cm
from mlxtend.plotting import plot_confusion_matrix

class TrainModel:
    def __init__(self, block, layers, image_channel, num_classes, num_epochs, batch_size, weights_path):
        self.block = block
        self.layers = layers
        self.image_channel = image_channel
        self.num_classes = num_classes
        self.epochs = num_epochs
        self.batch = batch_size
        self.path_to_weights = weights_path
        self.device = self._get_device()

    def _get_device(self):
        train_on_gpu = torch.cuda.is_available()
        if not train_on_gpu:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:0")
        return device

    def _get_default_transforms(self):
        # Note: Pytorch transforms works mostly on PIL Image so we convert it to that format and then apply
        # transformations. Also, normalize transforms should be applied when the images are converted to tensors.
        my_transforms = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.5),
                                            transforms.RandomCrop((224, 224)), transforms.RandomRotation(degrees=45), transforms.RandomVerticalFlip(p=0.2),
                                            transforms.ToTensor()])
        return my_transforms

    def plot_cnf_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        import itertools
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def start_training(self, path_to_train, path_to_valid, transformation=None, val_percent=0.2, lr_rate=1e-3):
        if transformation is None:
            train_dataset = DataProcessor(imgs_dir=path_to_train, transformations=self._get_default_transforms())
            valid_dataset = DataProcessor(imgs_dir=path_to_valid, transformations=self._get_default_transforms())
        else:
            train_dataset = DataProcessor(imgs_dir=path_to_train, transformations=transformation)
            valid_dataset = DataProcessor(imgs_dir=path_to_valid, transformations=transformation)

        print("="*40)
        print("Images for Training:", len(train_dataset))
        print("Images for Validation:", len(valid_dataset))
        print("="*40)
        trainloader = DataLoader(train_dataset, batch_size=self.batch, shuffle=True, drop_last=True)
        validloader = DataLoader(valid_dataset, batch_size=self.batch, shuffle=True, drop_last=True)
        """
        data = iter(trainloader).next()
        img, lbl = data["image"], data['label']
        print(img.shape)
        print(lbl)
        plt.imshow(img[10][0].numpy(), cmap='gray')
        plt.show()
        """
        # Instantiate model and other parameters
        model = ResNet(self.block, self.layers, self.image_channel, self.num_classes).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr_rate, weight_decay=1e-6)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
        # Varibles to track
        train_losses, val_losses, aucs = [], [], []
        metrics = {'accuracy': {0: [], 1: [], 2: []},
                   'sensitivity': {0: [], 1: [], 2: []},
                   'specificity': {0: [], 1: [], 2: []}
                   }
        valid_loss_min = np.Inf

        # Training loop
        for epoch in range(self.epochs):
            running_train_loss, running_val_loss = 0.0, 0.0
            epoch_loss = []
            # Put model on train mode
            model.train()
            for data in trainloader:
                images, labels = data['image'].to(self.device), data['label'].to(self.device, dtype=torch.long)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                running_train_loss += float(loss.item()) * images.size(0)
                epoch_loss.append(float(loss.item() * images.size(0)))

            scheduler.step(np.mean(epoch_loss))
            # Validation loop
            with torch.no_grad():
                model.eval()
                y_truth, y_prediction, scores = [], [], []
                for data in validloader:
                    images, labels = data['image'].to(self.device), data['label'].to(self.device, dtype=torch.long)
                    output = model(images)
                    loss = criterion(output, labels)
                    running_val_loss += float(loss.item()) * images.size(0)
                    output_pb = F.softmax(output.cpu(), dim=1)
                    top_ps, top_class = output_pb.topk(1, dim=1)
                    y_prediction.extend(list(top_class.flatten().numpy()))
                    y_truth.extend(list(labels.cpu().flatten().numpy()))
                    scores.extend(output_pb.numpy().tolist())

            # Calculate average losses
            avg_train_loss = running_train_loss / len(trainloader)
            avg_val_loss = running_val_loss / len(validloader)
            cnf_matrix = cm(y_truth, y_prediction, labels=[0, 1, 2])
            # Compute evaluations
            FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
            FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
            TP = np.diag(cnf_matrix)
            TN = cnf_matrix.sum() - (FP + FN + TP)
            # Convert to float
            f_p = FP.astype(float)
            f_n = FN.astype(float)
            t_p = TP.astype(float)
            t_n = TN.astype(float)

            # Calculate metrics
            accuracy = (t_p + t_n) / (f_p + f_n + t_p + t_n)
            recall_sensitivity = t_p / (t_p + f_n)
            specificity = t_n / (t_n + f_p)
            precision = t_p / (t_p + f_p)
            one_hot_true = np_utils.to_categorical(y_truth, num_classes=3)
            model_auc = roc_auc_score(one_hot_true, scores, average='weighted')

            # Append losses and track metrics
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            for index in range(3):
                metrics['accuracy'][index].append(accuracy[index])
                metrics['sensitivity'][index].append(recall_sensitivity[index])
                metrics['specificity'][index].append(specificity[index])
            aucs.append(model_auc)

            # Print results
            print("Epoch:{}/{} - Training Loss:{:.6f} | Validation Loss: {:.6f}".format(
                epoch + 1, self.epochs, avg_train_loss, avg_val_loss))
            print("Accuracy:{}\nPrecision:{}\nSensitivity:{}\nSpecificity:{}\nAUC:{}".format(
                accuracy, precision, recall_sensitivity, specificity, model_auc))

            # Save model
            if avg_val_loss <= valid_loss_min:
                print("Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(valid_loss_min, avg_val_loss))
                print("-" * 40)
                torch.save(model.state_dict(), "checkpoints/Resnet18_Covid_Viral_Normal.pth")
                # Update minimum loss
                valid_loss_min = avg_val_loss

        # Save plots
        plt.plot(train_losses, label='Training loss')
        plt.plot(val_losses, label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(frameon=False)
        plt.savefig('losses.png')
        plt.clf()

        plt.plot(metrics["accuracy"][0], label='Normal')
        plt.plot(metrics["accuracy"][1], label='Viral')
        plt.plot(metrics["accuracy"][2], label='Covid')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend(frameon=False)
        plt.savefig('accuracy.png')
        plt.clf()

        plt.plot(metrics["sensitivity"][0], label='Normal')
        plt.plot(metrics["sensitivity"][1], label='Viral')
        plt.plot(metrics["sensitivity"][2], label='Covid')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend(frameon=False)
        plt.savefig('sensitivity.png')
        plt.clf()

        plt.plot(metrics["specificity"][0], label='Normal')
        plt.plot(metrics["specificity"][1], label='Viral')
        plt.plot(metrics["specificity"][2], label='Covid')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend(frameon=False)
        plt.savefig('specificity.png')
        plt.clf()

        plt.plot(aucs, label='AUCs')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend(frameon=False)
        plt.savefig('aucs.png')
        plt.clf()

    def runtestset(self, path_to_images=None):
        if path_to_images:
            dataset = DataProcessor(imgs_dir=path_to_images, transformations=transforms.Compose([transforms.ToTensor()]))
            testloader = DataLoader(dataset, batch_size=self.batch, shuffle=True, drop_last=True)
            # Define model
            pass_one_batch = False
            model = ResNet(self.block, self.layers, self.image_channel, self.num_classes)
            criterion = nn.CrossEntropyLoss()
            # Load weights
            if self.path_to_weights:
                print("="*40)
                print("Model Weights Loaded")
                print("=" * 40)
                weights = torch.load(self.path_to_weights)
                model.load_state_dict(weights)
                model.to(self.device)

                # Make Predictions
                with torch.no_grad():
                    model.eval()
                    if pass_one_batch:
                        pass
                    else:
                        y_truth, y_prediction, scores = [], [], []
                        running_test_loss = 0.0
                        for data in testloader:
                            images, labels = data['image'].to(self.device), data['label'].to(self.device, dtype=torch.long)
                            output = model(images)
                            loss = criterion(output, labels)
                            running_test_loss += float(loss.item()) * images.size(0)
                            output_pb = F.softmax(output.cpu(), dim=1)
                            top_ps, top_class = output_pb.topk(1, dim=1)
                            y_prediction.extend(list(top_class.flatten().numpy()))
                            y_truth.extend(list(labels.cpu().flatten().numpy()))
                            scores.extend(output_pb.numpy().tolist())

                        # Computer metrics
                        avg_test_loss = running_test_loss / len(testloader)
                        cnf_matrix = cm(y_truth, y_prediction, labels=[0, 1, 2])
                        fig, ax = plot_confusion_matrix(conf_mat=cnf_matrix)
                        plt.show()
                        # Compute evaluations
                        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
                        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
                        TP = np.diag(cnf_matrix)
                        TN = cnf_matrix.sum() - (FP + FN + TP)
                        # Convert to float
                        f_p = FP.astype(float)
                        f_n = FN.astype(float)
                        t_p = TP.astype(float)
                        t_n = TN.astype(float)

                        # Calculate metrics
                        accuracy = (t_p + t_n) / (f_p + f_n + t_p + t_n)
                        recall_sensitivity = t_p / (t_p + f_n)
                        specificity = t_n / (t_n + f_p)
                        precision = t_p / (t_p + f_p)
                        one_hot_true = np_utils.to_categorical(y_truth, num_classes=3)
                        model_auc = roc_auc_score(one_hot_true, scores, average='weighted')

                        print("Test loss:{:.6f}".format(avg_test_loss))
                        print("Accuracy:{}\nPrecision:{}\nSensitivity:{}\nSpecificity:{}\nAUC:{}".format(accuracy,
                                                                                                         precision, recall_sensitivity, specificity, model_auc))

                        # Draw ROC Curve
                        fpr = dict()
                        tpr = dict()
                        roc_auc = dict()
                        # Computer FPR, TPR for two classes
                        scores = np.array(scores)
                        for i in range(3):
                            fpr[i], tpr[i], _ = roc_curve(one_hot_true[:, i], scores[:, i])
                            roc_auc[i] = auc(fpr[i], tpr[i])
                        """
                        # Computer Micro FPR, TPR
                        fpr["micro"], tpr["micro"], _ = roc_curve(one_hot_true_lbls.ravel(), y_scores.ravel())
                        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                        """
                        # Computer Macro FPR, TPR
                        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
                        # Then interpolate all ROC curves at this points
                        mean_tpr = np.zeros_like(all_fpr)
                        for i in range(3):
                            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
                        # Average and compute AUC
                        mean_tpr /= 3
                        fpr["macro"] = all_fpr
                        tpr["macro"] = mean_tpr
                        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

                        # Plot the ROC
                        plt.figure()
                        # Plot the micro-average
                        plt.plot(fpr["macro"], tpr["macro"], label='Average AUC Curve (area = {0:0.2f})'.format(roc_auc["macro"]),
                                 color='deeppink', linestyle=':', linewidth=4)
                        colors = cycle(['aqua', 'darkorange', 'teal'])
                        # Plot the different classes
                        for i, color in zip(range(3), colors):
                            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                                     label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

                        # Plot the figure
                        plt.plot([0, 1], [0, 1], 'k--', lw=2)  # plot the middle line
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.title('ROC - Malignancy')
                        plt.legend(loc="lower right")
                        plt.savefig('multiclass_auc.png')
                        plt.clf()
            else:
                print("Model Weights Required")


if __name__ == "__main__":
    # Hyper-param
    num_epcohs = 200
    batches = 30
    path_to_weights = None # Required when running model on test set
    train_images = "path_to_train_images\\"
    valid_images = "path_to_validation_images\\"
    test_images = "path_to_test_images\\"
    train_obj = TrainModel(ResidualBlock, [2, 2, 2, 2], 3, 3, num_epcohs, batches, path_to_weights)
    train_obj.start_training(train_images, valid_images)
    #train_obj.runtestset(test_images)
