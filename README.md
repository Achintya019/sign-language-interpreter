![A](https://github.com/user-attachments/assets/0799300f-5fa9-4abd-b925-a1eb1d537a69)
![Hello](https://github.com/user-attachments/assets/3a3cd9aa-c1b1-4233-a309-a2c8b7504323)
![ily](https://github.com/user-attachments/assets/367ca032-9ca7-4160-b7ae-51d9a5026730)
![y](https://github.com/user-attachments/assets/7055aec7-8b97-4fad-b7cb-8f34fd95b4ea)
![No](https://github.com/user-attachments/assets/8e0ff7ef-6d3d-44ba-bc43-5fdd268d0cfa)
![o](https://github.com/user-attachments/assets/1f709f3d-dfaf-42c4-9a81-b9f2fd16e739)

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


