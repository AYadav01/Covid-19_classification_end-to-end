from model_maskrcnn import MaskRCNN

path_to_weights = None # Required when doing predictions
path_to_images = "path_to_train_images\\"
path_to_masks = "path_to_train_masks\\"
path_to_test_images = "path_to_test_images\\"
path_to_test_masks = "path_to_test_masks\\"

if __name__ == "__main__":
    model_obj = MaskRCNN(path_to_dict=path_to_weights, num_classes=2)
    """
    Note: During training, validation set will be created from training set
    """
    model_obj.train_model(path_to_images, path_to_masks, num_epochs=100)
    #model_obj.runtestset(path_to_images, path_to_masks, pass_one_batch=False)
