import tensorflow as tf
import os

file_path = "D:\\develop\\PythonProjects\\MachineLearn\\out\\"

def linearRegressionByTf():
    """
    用tensorflow实现线性回归
    :return:
    """
    # 准备数据 x特征值，y目标值
    with tf.variable_scope("data_set"):  # 作用域，名称内不可有空格
        x = tf.random_normal([100, 1], mean=5.0, stddev=0.5, name="x_data")
        y_true = tf.matmul(x, [[0.7]]) + 0.8

    with tf.variable_scope("model_set"):
        # 建立模型，用变量定义参数，才能优化它
        weight = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0), name="weight")
        bias = tf.Variable(0.0, name="bias")

    with tf.variable_scope("predict_and_optimize"):
        # 预测
        y_predict = tf.matmul(x, weight) + bias

        # 建立损失函数
        loss = tf.reduce_mean(tf.square(y_true - y_predict))

        # 梯度下降优化损失 提供学习率，学习率越低，精度越高
        train_op = tf.train.GradientDescentOptimizer(0.03).minimize(loss)

    # 收集tensor，以标量图显示
    tf.summary.scalar(name="loss", tensor=loss)
    # 收集tensor，以直方图显示
    tf.summary.histogram("weight", weight)
    tf.summary.histogram("bias", bias)
    # 合并所有的收集情况
    merged = tf.summary.merge_all()

    # 保存与加载模型
    saver = tf.train.Saver()

    # 初始化变量op
    init_op = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init_op)
        print("x:", x.eval())
        print("y:", y_true.eval())
        print("初始化的权重：", weight.eval(), "，偏移量：", bias.eval())
        writer = tf.summary.FileWriter("D:\\develop\\PythonProjects\\MachineLearn", train_op.graph)
        if os.path.exists(file_path + "checkpoint"):  # 检查checkpoint文件是否存在
            print("采用上次训练的模型")
            saver.restore(session, file_path + "model")  # 加载model文件
        for i in range(500):
            session.run(train_op)
            # print("y_predict:", y_predict.eval())
            print("优化后的权重：%f" % (weight.eval()),
                  "，偏移量：%f" % (bias.eval()), "，误差：%f" % (loss.eval()))
            # 每训练一次，收集一次数据
            writer.add_summary(summary=session.run(merged), global_step=i)
        saver.save(session, file_path + "model")  # 保存为model文件



if __name__ == '__main__':
    linearRegressionByTf()