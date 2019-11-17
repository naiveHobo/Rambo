# Rambo
Implementation of Rambo from the second challenge for the Udacity Self-Driving Car initiative with the option to visualize the activation maps to see what your model learns.

![Demo](demo.gif)

## Instructions
Run the preprocessing script on the [data](https://github.com/udacity/self-driving-car/blob/master/datasets/CH2/Ch2_002.tar.gz.torrent) provided by Udacity:
```
python3 dataset.py --y angle
```

To get more details about running the training/inference code:
```
python3 main.py -h
```

#### Training
Train the model using the following command:
```
python3 main.py --mode train
```

Add the **--visualize** argument to the above command to build the activation map visualization graph as well.

#### Inference
You can run the trained model on your testing data using the following command:
```
python3 main.py --mode test
```
