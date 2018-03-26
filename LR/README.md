# 逻辑回归分类器

## mnist

mnist手写体识别是一个经典的问题。TensorFlow官方例子中给出的是使用CNN网络实现识别。此处参照TensorFlow官方例子和[TensorFlow实现逻辑回归分类器](https://blog.csdn.net/diligent_321/article/details/52937400)给出一个使用传统机器学习算法“逻辑回归”实现识别的demo。

### 代码说明

逻辑回归可以简单的认为是一个y=W·x+b的问题。目标是基于一个最小化指标求解W和b。

mnist原始数据集的每张图片由28 * 28个像素点构成，每个像素点用一个灰度值表示。计算时将28 * 28个像素转换为一个一维的向量。因此每条样本相当于一个维数为784的向量。并且mnist的手写体是0-9十个数字，其label是一个10 * 1的向量，比如0对应的向量是[1,0,0,0,0,0,0,0,0,0]。因此求解的W是一个784 * 10的矩阵，b是一个1 * 10的向量。

TensorBoard 通过读取 TensorFlow 的事件文件来运行。如果希望通过TensorBoard理解、调试、优化TensorFlow，需要在代码中生成汇总数据（Summary data）。TensorBoard各个面板说明：

![](../images/tensorboard.jpg)

mnist_without_namescope.py文件只是在实现逻辑回归分类的基础上仅仅汇总了graph，并没有在代码中为变量名划定范围优化可视化效果。其生成的graph难以帮助用户理解构建的计算流程。

![](../images/lr_mnist_without_namescope.jpg)

mnist_with_namescope.py文件使用namescope添加名称和作用域优化graph的呈现。并且除了计算图以外收集了一些准确率、损失等标量信息绘制scalars面板下的折线图，同时收集了诸如权重等信息绘制distrubitions,histograms面板下的图。

![](../images/lr_mnist_with_namescope.jpg)

### 参考文档：

[详解 MNIST 数据集](https://blog.csdn.net/simple_the_best/article/details/75267863)

[TensorBoard:可视化学习](http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/summaries_and_tensorboard.html)

[TensorBoard:图表可视化](http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/graph_viz.html)

[TENSORFLOW - TENSORBOARD可视化](https://gaussic.github.io/2017/08/16/tensorflow-tensorboard/)
