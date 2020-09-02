import pandas as pd
import torch.utils.data as data_utils
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model_mlp import MLP
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, \
    roc_curve, roc_auc_score, auc
import torch.optim as optim
from keras.utils import np_utils
from itertools import cycle
from scipy import interp
from sklearn.metrics import confusion_matrix as cm
from mlxtend.plotting import plot_confusion_matrix
import torch

class TrainModel:
    def __init__(self, input_filter, hidden_neurons, num_epochs, batch_size, weights_path):
        self.input_neuron = input_filter
        self.hidden_layers = hidden_neurons
        self.epochs = num_epochs
        self.batch = batch_size
        self.thresh = 0.5
        self.path_to_weights = weights_path
        self.device = self._get_device()

    def _get_device(self):
        train_on_gpu = torch.cuda.is_available()
        if not train_on_gpu:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:0")
        return device

    def start_training(self, path_to_csv):
        df = pd.read_csv(path_to_csv)
        shuffled_df = df.sample(frac=1)
        # Get numpy data from csv
        data_y = shuffled_df.iloc[:, -1].to_numpy()
        data_x = shuffled_df.drop(['slice_id', 'label'], axis=1).to_numpy()
        # Split the dataset
        x_train, x_valid, y_train, y_valid = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
        # Standarize the training data
        sc = StandardScaler()
        x_train, x_valid = sc.fit_transform(x_train), sc.fit_transform(x_valid)
        # Covert to tensors
        x_train, y_train = torch.from_numpy(x_train), torch.from_numpy(y_train)
        x_valid, y_valid = torch.from_numpy(x_valid), torch.from_numpy(y_valid)
        # Create datasets
        train = data_utils.TensorDataset(x_train.float(), y_train)
        validation = data_utils.TensorDataset(x_valid.float(), y_valid)
        trainloader = data_utils.DataLoader(train, batch_size=self.batch, shuffle=True, drop_last=True)
        validloader = data_utils.DataLoader(validation, batch_size=self.batch, shuffle=True, drop_last=True)

        # Instantiate model and other parameters
        model = MLP(self.input_neuron, self.hidden_layers).to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
        # Varibles to track
        train_losses, val_losses, aucs = [], [], []
        valid_loss_min = np.Inf
        # Training loop
        metrics = {'accuracy': {0: [], 1: [], 2: []},
                   'sensitivity': {0: [], 1: [], 2: []},
                   'specificity': {0: [], 1: [], 2: []}
                   }

        for epoch in range(self.epochs):
            running_train_loss, running_val_loss = 0.0, 0.0
            epoch_loss = []
            model.train()
            for images, labels in trainloader:
                images, labels = images.to(self.device), labels.to(self.device, dtype=torch.long)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                #running_train_loss += loss.detach().item()
                running_train_loss += float(loss.item()) * images.size(0)
                epoch_loss.append(float(loss.item() * images.size(0)))

            scheduler.step(np.mean(epoch_loss))
            # Validation loop
            with torch.no_grad():
                model.eval()
                y_truth, y_prediction, scores = [], [], []
                for images, labels in validloader:
                    images, labels = images.to(self.device), labels.to(self.device, dtype=torch.long)
                    output = model(images)
                    loss = criterion(output, labels)
                    running_val_loss += float(loss.item()) * images.size(0)
                    output_pb = F.softmax(output.cpu(), dim=1)
                    top_ps, top_class = output_pb.topk(1, dim=1)
                    y_prediction.extend(list(top_class.flatten().numpy()))
                    y_truth.extend(list(labels.cpu().flatten().numpy()))
                    scores.extend(output_pb.numpy().tolist())

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

            # Compute metrics
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
            print("Epoch:{}/{} - Training Loss:{:.6f} | Validation Loss: {:.6f}".format(
                epoch + 1, self.epochs, avg_train_loss, avg_val_loss))
            print("Accuracy:{}\nPrecision:{}\nSensitivity:{}\nSpecificity:{}\nAUC:{}".format(
                accuracy, precision, recall_sensitivity, specificity, model_auc))

            # Save model
            if avg_val_loss <= valid_loss_min:
                print("Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(valid_loss_min,
                                                                                                avg_val_loss))
                print("-" * 40)
                torch.save(model.state_dict(), "..\\checkpoints\\MLP_Covid_Viral_Normal.pth")
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


    def run_testset(self, path_to_csv):
        df = pd.read_csv(path_to_csv)
        shuffled_df = df.sample(frac=1)
        # Get numpy data from csv
        data_y = shuffled_df.iloc[:, -1].to_numpy()
        data_x = shuffled_df.drop(['slice_id', 'label'], axis=1).to_numpy()
        # Standarize the test data
        sc = StandardScaler()
        data_x = sc.fit_transform(data_x)

        # Covert to tensors
        x_test, y_test = torch.from_numpy(data_x), torch.from_numpy(data_y)
        # Create datasets
        test = data_utils.TensorDataset(x_test.float(), y_test)
        testloader = data_utils.DataLoader(test, batch_size=self.batch, shuffle=True, drop_last=True)

        # Instantiate model and other parameters
        model = MLP(self.input_neuron, self.hidden_layers)
        if self.path_to_weights:
            print("=" * 40)
            print("Model Weights Loaded")
            print("=" * 40)
            weights = torch.load(self.path_to_weights)
            model.load_state_dict(weights)
            model.to(self.device)

        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            model.eval()
            y_truth, y_prediction, scores = [], [], []
            running_test_loss = 0.0
            for images, labels in testloader:
                images, labels = images.to(self.device), labels.to(self.device, dtype=torch.long)
                output = model(images)
                loss = criterion(output, labels)
                running_test_loss += float(loss.item()) * images.size(0)
                output_pb = F.softmax(output.cpu(), dim=1)
                top_ps, top_class = output_pb.topk(1, dim=1)
                y_prediction.extend(list(top_class.flatten().numpy()))
                y_truth.extend(list(labels.cpu().flatten().numpy()))
                scores.extend(output_pb.numpy().tolist())

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

            # Compute metrics
            accuracy = (t_p + t_n) / (f_p + f_n + t_p + t_n)
            recall_sensitivity = t_p / (t_p + f_n)
            specificity = t_n / (t_n + f_p)
            precision = t_p / (t_p + f_p)
            one_hot_true = np_utils.to_categorical(y_truth, num_classes=3)
            model_auc = roc_auc_score(one_hot_true, scores, average='weighted')

            print("Test loss:{:.6f}".format(avg_test_loss))
            print("Accuracy:{}\nPrecision:{}\nSensitivity:{}\nSpecificity:{}\nAUC:{}".format(accuracy,
                                                                                             precision,
                                                                                             recall_sensitivity,
                                                                                             specificity, model_auc))

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


if __name__ == "__main__":
    # Hyper-param
    in_channel = 21
    hidden_outputs = [128, 64, 32, 3]
    num_epcohs = 500
    batches = 50
    path_to_weights = None # Required during predictions
    train_csv = "path_to_train.csv"
    test_csv = "path_to_test.csv"
    train_obj = TrainModel(in_channel, hidden_outputs, num_epcohs, batches, path_to_weights)
    """
    Note: validation set is created out of training set
    """
    train_obj.start_training(train_csv)
    #train_obj.run_testset(test_csv)
