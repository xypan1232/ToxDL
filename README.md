# ToxDL: Deep learning using primary structure and domain embeddings for assessing protein toxicity
<br>
The process of developing genetically engineered (GE) food crops involves introducing proteins from one species to a crop plant species or to modify an existing protein to improve agricultural important traits such as yield, pest resistance and herbicide tolerance. or both research and regulation purpose it is crucial to examine/assess the potential allergenicity and toxicity of the introduced or gene-edited protein to ensure the food and environment safety of the crop products.<br>
In this study, we develop an interpretable deep learning-based method, ToxDL, to classify toxic proteins from non-toxic proteins using sequences alone. There are two main components in the multi-modal ToxDL. The first component is based on CNNs, in which sequences are encoded in one-hot matrix, which is fed into a CNN. The second component is a multilayer perceptron with domain information. The domains are first scanned from protein sequences using InterProscan. Instead of using high-dimensional one-hot encoding, domains of proteins are encoded in embeddings learned by word2vec, which is fed into a fully connected layer together with feature maps from the CNN.
<br>
<br>

# Dependency:
python              3.5 <br>
TensorFlow          1.12.0 <br>
tensorboard         1.12.2 <br>
scikit-learn         0.21.3 <br>
scipy               1.3.1 <br>
numpy               1.17.1 <br>
pandas              0.24.2 <br>
joblib              0.13.2 <br>
gensim              3.8.0 <br>
logomaker           0.8 <br>
matplotlib          3.0.3 <br>


# OS Requirements

This package is supported for *Linux* operating systems. The package has been tested on the following systems:

Linux: Ubuntu 16.04  

# Demo
You can directly  run the commend to run ToxDL model on animal training, validaiton and test set by running: <br>
``` python ToxDL.py``` 
<br>
The above command will output performance metrics in F1-score, MCC, auROC and auPRC for 10 running and the average performance of ToxDL for the 10 running. <br>
You can also specity the hyperparamters by modifying the file "TestFiles/000_test.test"<br>
The trained model is saved at the directory parameters/. <br>
The output file is saved at the directory predictions/. <br>

# Web service
You can also predict toxicity score for new proteins using the online web service at http://www.csbio.sjtu.edu.cn/bioinf/ToxDL/. <br>
