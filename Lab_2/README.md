# Lab 2. CNN on NumPy (MNIST)

## Task
Create a convolutional network for classifying handwritten digits ([MNIST](http://yann.lecun.com/exdb/mnist)).

Criteria:
* The number of layers, convolution dimension and convolution and pooling hyperparameters are at your discretion.
* Train size - 70%, test size - 30%
* Implement calculation of the confusion matrix, which should be used to calculate accuracy, precision, recall, F-measure, plot ROC-curve, calculate AUC (for each class!). 
* Preliminarily build t-SNE and evaluate the reasons for misclassification.

---
## Results 

The final network has the following architecture
```jupyter
# Feature extractor
CNN.add(Conv2d(in_channels=1, out_channels=2, kernel_size=3, random_seed=seed))  # 2 x 26 x 26
CNN.add(ReLU())
CNN.add(AvgPool2d(kernel_size=2))  # 2 x 13 x 13
CNN.add(Conv2d(in_channels=2, out_channels=3, kernel_size=3, random_seed=seed))  # 3 x 11 x 11
CNN.add(ReLU())
CNN.add(AvgPool2d(kernel_size=2))  # 3 x 5 x 5
CNN.add(Flatten())
# Head
CNN.add(Linear(in_neurons=3 * 5 * 5, out_neurons=32, random_seed=seed))
CNN.add(ReLU())
CNN.add(Linear(in_neurons=32, out_neurons=num_classes, random_seed=seed))
CNN.add(Softmax())
```
### Loss
![train_test_loss.png](img%2Ftrain_test_loss.png)

### Classification metrics
![classification_report.png](img%2Fclassification_report.png)

### Confusion matrix
![confusion_matrix.png](img%2Fconfusion_matrix.png)

### ROC curves
![roc_curves.png](img%2Froc_curves.png)

## Sources (Russian)
1. [Свёрточные нейросети (Яндекс учебник)](https://education.yandex.ru/handbook/ml/article/svyortochnye-nejroseti)
2. [Свёрточная нейронная сеть с нуля ](https://programforyou.ru/poleznoe/convolutional-network-from-scratch-part-zero-introduction)
## Sources (English)
1. [Backpropagation in Convolutional Neural Networks (Youtube)](https://www.youtube.com/watch?v=z9hJzduHToc&list=WL&index=7&t=4s)
2. [Backpropagation in CNNs](https://youtu.be/pUCCd2-17vI)
3. [The Softmax function and its derivative](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)

## Comments for myself
* The convolution kernels are just a set of tensors. These tensors have the same size, and their number determines the depth of the output array. At the same time, the depth of the filters themselves coincides with the number of channels of the input image.
* The gradients on the weights of the convolution layer are computed as the convolution of the gradients of the output feature maps with the input data.
* The gradients to be passed to the previous layer are obtained by convolving the gradients of the output feature maps with the transposed filter weights. This operation is also called transposed convolution or cross-correlation.
* Maxpooling selects the maximum value from each window (kernel) of the input data. During back propagation, the gradient is propagated only through the element that was selected as the maximum.
* Mean pooling calculates the average value of the elements in each window (kernel) of the input data. During back propagation, the gradient is evenly distributed among all elements in the window.
