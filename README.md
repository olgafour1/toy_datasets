# Neighbor-based-Multiple-Instance-Learning
The code can be used to run experiments on the the COLON cancer and UCSB datasets. 
To test the siamese version of our model the weights are stored 
in the root directory under the name colon_weights and cell_weights, respectively. 
One can select between two different running modes: euclidean and siamese one. 
Using the argument k, the number of neighbors can be specified. 
Because the running time grows proportionally to the number of K, it is advised to tested it for K<10.

## Installation
It is advised to use a conda installation, as it is more straightforward.
 - python=3.7
 - tensorflow-gpu=2.1
 - scikit-learn 
 - Pillow
 - opencv

## How to use
The script uses a number of command line arguments to work the most important of which are the following :

* k- the number of neighbors taken into account
* data_path - the directory where the images are stored
* siamese_weights_path - the directory where the weights of the pre-trained siamese network are stored
* extention - the file extention of the image patches
* mode - (euclidean or siamese) version of the model 
* weight_file - define whether there is, or not a file of weights
* experiment_name - the name of the experiment to run
* input_shape - (w,h,d) of the image patches
* folds - the number of folds to be used in the k-cross fold validation: 10 for the COLON cancer dataset, 4 for the UCSB



For the COLON cancer dataset the script can be executed from the command line typing the command:


```sh
cd <root_directory>
$ python run.py --experiment_name colon_3 --mode siamese --k 5 --data_path ColonCancer --input_shape 27 27 3 --extention bmp --siamese_weights_path colon_siamese_weights --siam_pixel_distance 20 --data colon
```
or the following command in case the euclidean mode is selected:
```sh
cd <root_directory>
$ python run.py --experiment_name colon_3 --mode euclidean -k 5 --data_path ColonCancer --input_shape 27 27 3 --extention bmp --data colon
```


For the UCSB cancer dataset the script can be executed from the command line typing the command:
```sh
cd <root_directory>
$ python run.py --arch '{"type": "Conv2D", "channels": 36, "kernel": (4, 4)},{"type": "MaxPooling2D", "pool_size": (2, 2)},{"type": "Conv2D", "channels": 48, "kernel": (3, 3)},{"type": "MaxPooling2D", "pool_size": (2, 2)},{"type": "Flatten"},{"type": "relu", "size": 512},{"type": "Dropout", "rate": 0.2},{"type": "relu", "size": 512},{"type": "Dropout", "rate": 0.2}' --k 3 --folds 4 --data ucsb --input_shape 32 32 3  --mode siamese  --data_path Breast_Cancer_Cells --ext tif --experiment_name ucsb_5 --siam_pixel_distance 30 --siamese_weights_path ucsb_siamese_weights

```
or the following command in case the euclidean mode is selected:
```sh
cd <root_directory>
$ python run.py --arch '{"type": "Conv2D", "channels": 36, "kernel": (4, 4)},{"type": "MaxPooling2D", "pool_size": (2, 2)},{"type": "Conv2D", "channels": 48, "kernel": (3, 3)},{"type": "MaxPooling2D", "pool_size": (2, 2)},{"type": "Flatten"},{"type": "relu", "size": 512},{"type": "Dropout", "rate": 0.2},{"type": "relu", "size": 512},{"type": "Dropout", "rate": 0.2}' --k 3 --folds 4 --data ucsb --input_shape 32 32 3  --mode euclidean  --data_path Breast_Cancer_Cells --ext tif --experiment_name ucsb_5 


```
# Neighbor-based-Multiple-Instance-Learning
