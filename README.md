# fashion-items-classification-and-compatibility

Data transformation steps

1. I wrote a python script to create 2 .txt files (train_compatibility.txt and valid_compatibility.txt) where I stored pairs of image ids in each line and the corresponding labels in this order:
label image1_id image2_id

2. I added a function (create_compatibility) in the class polyvore_dataset.
It returns X_compatTrain, X_compatValid, Y_compatTrain, Y_compatValid. The X_* contain tuples of the 2 input images (image1.jpg,image2.jpg) and Y_* contain labels.

3. I changed the getitem functions in classes polyvore_train and polyvore_test so that they returned 2 images (with original transforms applied) and their label.

4. I made another function get_dataloader_compat to incorporate the above changes to get data.
