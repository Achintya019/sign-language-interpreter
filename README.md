# Sign-Language-Interpreter
* This project can successfully classify 30 signs according to American sign language using Random forest classifier and a simple Keras deep learning model

## About the dataset
* The dataset consists of 3000 images, 100 images for each symbol.
* Images contain 26 letters, and 4 phrases labeled through 0-29
* The final training dataset is constructed as follows:
 ``` 
  data/
├── 0/
│ ├── image1.jpg
│ ├── image2.jpg
│ ├── ...
│ └── image100.jpg
├── 1/
│ ├── image1.jpg
│ ├── image2.jpg
│ ├── ...
│ └── image100.jpg
├── 2/
│ ├── image1.jpg
│ ├── image2.jpg
│ ├── ...
│ └── image100.jpg
├── ...
├── 28/
│ ├── image1.jpg
│ ├── image2.jpg
│ ├── ...
│ └── image100.jpg
└── 29/
├── image1.jpg
├── image2.jpg
├── ...
└── image100.jpg
 ```
## Create Environment and installation of packages
* Open Terminal in Linux
* conda create --name myenv python=3.9
* conda activate myenv
* pip install -r requirements.txt

## Execute Format 
* open terminal in Linux
* mkdir sign-language-interpreter
* cd sign-language-interpreter
* gitclone https://github.com/Achintya019/sign-language-interpreter.git

## Inference on Live Camera
### Tensorflow model
* python3 tensorflow_inference.py
### Random Forest Classifier 
* python3 inference_classifier.py


