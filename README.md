# Machine-Learning-Summer-Code
Repository containing code for my learning of classical and quantum machine learning with help from the QuDOS group at the University of Exeter. Classical models include a convolutional neural network for classification of the MNIST dataset using TensorFlow. Quantum models include a variational quantum classifier, quantum convolutional neural network and a fully connected quantum neural network using the [Yao Quantum](https://yaoquantum.org/) package for Julia with a goal to compare performance of the three QML architectures with multiple simple datasets.

## Models


* Variational Quantum Classifier

![vqc](images/VQC_Circuit.png)

The simplest of the three architectures and an essential base for the following models is the VQC, it includes a feature map to encode classical data using angle encoding. The ansatz is a circuit with variational parameters that can be updated with angles determined by a classical optimization process. Using a variational circuit from [YaoExtensions](https://github.com/QuantumBFS/YaoExtensions.jl) the ansatz is composed of multiple layers of parametrized rotations on each qubit followed by a ladder of CNOTs and a final series of parametrized rotations. The number of layers can be changed to alter the number of parameters the circuit uses. Each qubit is then measured.

* Quantum Convolutional Neural Network

![qcnn](images/QCNN.png)


* Fully Connected Neural Network with no Pooling Layers

![fc](images/fully_connected.png)
