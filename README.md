# Machine-Learning-Summer-Code
Repository containing code for my learning of classical and quantum machine learning with help from the QuDOS group at the University of Exeter.

Classical models include a convolutional neural network for classification of the MNIST dataset using TensorFlow. Quantum models include a quantum kernel with support vector classification in Qiskit, a variational quantum classifier, quantum convolutional neural network and a fully connected quantum neural network using the [Yao Quantum](https://yaoquantum.org/) package for Julia with a goal to compare performance of the three QML architectures with multiple simple datasets.

## Models


* Variational Quantum Classifier

<img src="images/VQC_Circuit.png" height="220">

The simplest of the three architectures and an essential base for the following models is the VQC, it includes a feature map to encode classical data using angle encoding. The ansatz is a circuit with variational parameters that can be updated with angles determined by a classical optimization process. Using a variational circuit from [YaoExtensions](https://github.com/QuantumBFS/YaoExtensions.jl) the ansatz is composed of multiple layers of parametrized rotations on each qubit followed by a ladder of CNOTs and a final series of parametrized rotations. The number of layers can be changed to alter the number of parameters the circuit uses. Each qubit is then measured.

* Quantum Convolutional Neural Network

![qcnn](images/QCNN.png)

QCNN architecture is comprised of convolutional layers followed by pooling layers much like the classical counterpart. The convolutional layers contain two rows of two qubit unitaries which act on alternating pairs of qubits, composed of variational parameters. These unitaries are similar to that of the VQC but instead use a slightly different architecture that only acts on 2 neighbouring qubits.

The pooling layer contains measurements on half of the qubits with the outcome of the measurement controlling a unitary acting on the neighbouring qubit, in which a pooling operation of two qubits maps the 2 qubit Hilbert space to a 1 qubit Hilbert space. Each pooling layer is followed by a convolutional layer until the final Hilbert space of the system is sufficiently small and the output state of the circuit is measured by an operator.

In theory, the QCNN should have the best performance compared to that of the VQC and the fully connected models and should be close to that of a classical CNN.

The architecture used in the QCNN follows ideas from [arXiv:1810.03787v2](https://arxiv.org/pdf/1810.03787.pdf), [arXiv:2011.02966v1](https://arxiv.org/pdf/2011.02966.pdf), [arXiv:2108.00661v1](https://arxiv.org/pdf/2108.00661.pdf) and [arXiv:2009.09423v1](https://arxiv.org/pdf/2009.09423.pdf).


* Fully Connected Neural Network with no Pooling Layers

![fc](images/fully_connected.png)

The fully connected model is essentially a QCNN without pooling layers and is instead completely composed of layers of the same two qubit unitaries as used in the QCNN, therefore keeping a constant input dimension throughtout the entire circuit.

## Results

The datasets used to benchmark the models can be found in the [datasets](datasets/) folder and include 2 feature, 3 feature, 5 feature and circular datasets.
