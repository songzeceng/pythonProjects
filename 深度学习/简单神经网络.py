import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def full_connected(train=True):
    # 读取mnist数据集，采取one_hot编码
    mnist = input_data.read_data_sets("D:\\develop\\PythonProjects\\MachineLearn\\mnist", one_hot=True)

    # 收集数据，采用占位符
    with tf.variable_scope("data"):
        x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        y_true = tf.placeholder(dtype=tf.int32, shape=[None, 10])

    # 构造模型，输出预测值
    with tf.variable_scope("model"):
        # 随机化权重和偏移量
        weight = tf.Variable(tf.random_normal([784, 10], mean=0.0, stddev=1.0), name="weight")
        bias = tf.Variable(tf.constant(1.0, shape=[10]))

        y_predict = tf.matmul(x, weight) + bias

    # 计算误差均值
    with tf.variable_scope("softmax"):
        # 求平均交叉熵损失
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

    # 梯度下降优化
    with tf.variable_scope("optimize"):
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 计算准确度
    with tf.variable_scope("accuracy"):
        equal_list = tf.equal(tf.arg_max(y_true, 1), tf.arg_max(y_predict, 1))
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 收集变量
    tf.summary.scalar("losses", loss)
    tf.summary.scalar("accuracy", accuracy)
    tf.summary.histogram("weight", weight)
    tf.summary.histogram("bias", bias)

    merged = tf.summary.merge_all()

    # 保存模型
    saver = tf.train.Saver()

    # 初始化变量
    init_op = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init_op)

        if train:
            fileWriter = tf.summary.FileWriter("D:\\develop\\PythonProjects\\MachineLearn\\events", train_op.graph)

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
                # 收集变量
                fileWriter.add_summary(summary=session.run(merged, feed_dict={
                    x: mnist_x,
                    y_true: mnist_y
                }), global_step=i)

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
    full_connected(train=False)