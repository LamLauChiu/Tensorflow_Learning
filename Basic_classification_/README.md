https://medium.com/%E9%9B%9E%E9%9B%9E%E8%88%87%E5%85%94%E5%85%94%E7%9A%84%E5%B7%A5%E7%A8%8B%E4%B8%96%E7%95%8C/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-ml-note-convolution-neural-network-%E5%8D%B7%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF-bfa8566744e9
https://zhuanlan.zhihu.com/p/46214424

Convolutional neural network (CNN)

Convolution Layers (卷積層)

Tensorflow:

tf.nn.conv2d( input, 
              filter, 
              strides, 
              padding, 
              use_cudnn_on_gpu=None, 
              data_format=None, 
              name=None)

** zero padding ,stride
「padding = ‘VALID’」
「padding = ‘SAME’」
 
https://medium.com/@chih.sheng.huang821/%E5%8D%B7%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF-convolutional-neural-network-cnn-%E5%8D%B7%E7%A9%8D%E8%A8%88%E7%AE%97%E4%B8%AD%E7%9A%84%E6%AD%A5%E4%BC%90-stride-%E5%92%8C%E5%A1%AB%E5%85%85-padding-94449e638e82


Feature map
★Feature map width=[(Original width-Kernel width)/(Stride+1)]+1


Subsampling/ Pooling Layer(采樣/池化層)

https://medium.com/%E9%9B%9E%E9%9B%9E%E8%88%87%E5%85%94%E5%85%94%E7%9A%84%E5%B7%A5%E7%A8%8B%E4%B8%96%E7%95%8C/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-ml-note-overfitting-%E9%81%8E%E5%BA%A6%E5%AD%B8%E7%BF%92-6196902481bb
什麼是Overfitting(過度學習)

https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html
Underfitting vs. Overfitting


model.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

Optimizer :
1. SGD-準確率梯度下降法 (stochastic gradient decent)
2. Momentum
3. AdaGrad
4. Adam (目前較常使用的Optimizer)



Concept of Deep Learning : Deep Neural Network (DNN) & Backpropagation 

https://zhuanlan.zhihu.com/p/25794795

Deep Learning

/////

VOC Dataset
https://blog.csdn.net/zhangjunbob/article/details/52769381

COCO Dataset
https://zhuanlan.zhihu.com/p/29393415

LabelImg 
= A graphical image annotation tool and label object bounding boxes in images 
https://github.com/tzutalin/labelImg

Object detection
https://medium.com/%E9%9B%9E%E9%9B%9E%E8%88%87%E5%85%94%E5%85%94%E7%9A%84%E5%B7%A5%E7%A8%8B%E4%B8%96%E7%95%8C/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-ml-note-yolo-%E5%88%A9%E7%94%A8%E5%BD%B1%E5%83%8F%E8%BE%A8%E8%AD%98%E5%81%9A%E7%89%A9%E4%BB%B6%E5%81%B5%E6%B8%AC-object-detection-%E7%9A%84%E6%8A%80%E8%A1%93-3ad34a4cac70

//////

CatBoost、LightGBM、XGBoost，這些演算法你都瞭解嗎？
https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/510420/

////

Feature Engineering 特徵工程中常見的方法
https://vinta.ws/code/feature-engineering.html

////

Ensemble Learning(集成學習)
https://rpubs.com/skydome20/R-Note16-Ensemble_Learning

