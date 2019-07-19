import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def init_weight(shape):
    return tf.Variable(tf.random_normal(shape, mean=0.0, stddev=1.0), name="weight")


def init_bias(shape):
    return tf.Variable(tf.constant(1.0, shape=shape))


def model():
    # 收集数据，采用占位符
    with tf.variable_scope("data"):
        x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        y_true = tf.placeholder(dtype=tf.int32, shape=[None, 10])

    # 第一层神经网络
    with tf.variable_scope("layer1"):
        # 初始化权重和偏移
        w_layer1 = init_weight(shape=[5, 5, 1, 32])
        b_layer1 = init_bias(shape=[32])

        # 改变特征值形状 [None, 748] => [None, 28, 28, 1]
        x_reshape = tf.reshape(x, [-1, 28, 28, 1])
        # 卷积、激活，给定卷积步长
        x_relu = tf.nn.relu(tf.nn.conv2d(x_reshape, w_layer1, strides=[1, 1, 1, 1], padding="SAME")
                            + b_layer1)
        # 池化
        x_pool = tf.nn.max_pool(x_relu, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding="SAME")

    # 第二层神经网络
    with tf.variable_scope("layer2"):
        w_layer2 = init_weight(shape=[5, 5, 32, 64])
        b_layer2 = init_bias(shape=[64])

        x_relu = tf.nn.relu(tf.nn.conv2d(x_pool, w_layer2, strides=[1, 1, 1, 1], padding="SAME")
                         + b_layer2)

        x_pool = tf.nn.max_pool(x_relu, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding="SAME")

    # 全连接层
    with tf.variable_scope("full_connection"):
        w_fc = init_weight(shape=[7 * 7 * 64, 10])
        b_fc = init_bias(shape=[10])

        x_reshape = tf.reshape(x_pool, [-1, 7 * 7 * 64])

        y_predict = tf.matmul(x_reshape, w_fc) + b_fc

    return x, y_true, y_predict


def cons_cv(train):
    # 读取mnist数据集，采取one_hot编码
    mnist = input_data.read_data_sets("D:\\develop\\PythonProjects\\MachineLearn\\mnist", one_hot=True)

    # 神经网络获取特征值、真实值和预测值
    x, y_true, y_predict = model()

    # 计算误差均值
    with tf.variable_scope("softmax"):
        # 求平均交叉熵损失
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

    # 梯度下降优化
    with tf.variable_scope("optimize"):
        train_op = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)

    # 计算准确度
    with tf.variable_scope("accuracy"):
        equal_list = tf.equal(tf.arg_max(y_true, 1), tf.arg_max(y_predict, 1))
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 保存模型
    saver = tf.train.Saver()

    # 初始化变量
    init_op = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init_op)

        if train:

            # 训练2000次
            for i in range(2000):
                # 每次取50张图片训练
                mnist_x, mnist_y = mnist.train.next_batch(50)
                # 训练，给出特征值和目标值
                session.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})
                print("第", (i + 1), "次训练，准确率为：", session.run(accuracy, feed_dict={
                    x: mnist_x,
                    y_true: mnist_y
                }))

            # 保存模型
            saver.save(session,
                       save_path="D:\\develop\\PythonProjects\\MachineLearn\\out\\picturePredict")
        else:
            # 预测，先读取模型
            saver.restore(session, "D:\\develop\\PythonProjects\\MachineLearn\\out\\picturePredict")

            # 预测100次
            for i in range(100):
                # 每次预测1张图片，先读取测试集的特征值和目标值
                x_test, y_test = mnist.train.next_batch(1)

                # 预测
                print("第", (i + 1), "张图片，目标值：", tf.arg_max(y_test, 1).eval(), "，预测值：",
                      tf.arg_max(session.run(y_predict, feed_dict={
                          x: x_test,
                          y_true: y_test
                      }), 1).eval())

    return None


if __name__ == '__main__':
    cons_cv(train=True)