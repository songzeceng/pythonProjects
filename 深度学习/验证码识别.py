import tensorflow as tf


def init_weight(shape):
    return tf.Variable(tf.random_normal(shape, mean=0.0, stddev=1.0), name="w")


def init_bias(shape):
    return tf.Variable(tf.random_normal(shape, mean=0.0, stddev=1.0), name="b")


def read_and_decode(file_path):
    """
    读取并解码验证码目录
    :param file_path: 目录路径
    :return: 图片特征值和目标值的批处理op
    """
    file_queue = tf.train.string_input_producer([file_path])
    reader = tf.TFRecordReader()

    key, value = reader.read(file_queue)

    features = tf.parse_single_example(value, features={
        "image": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.string)
    })

    image = tf.reshape(tf.decode_raw(features["image"], tf.uint8), [20, 80, 3])
    label = tf.reshape(tf.decode_raw(features["label"], tf.uint8), [4])

    return tf.train.batch([image, label], batch_size=10, num_threads=1, capacity=10)


def one_hot_encode(label):
    """
    将目标值进行onehot编码
    :param label: 待编码的目标值
    :return: 编码后的目标值
    """
    return tf.one_hot(label, depth=26, on_value=1.0, axis=2)



def fc_model(image):
    """
    根据特征值预测
    :param image: 特征值
    :return: 预测值
    """
    with tf.variable_scope("model"):
        weight = init_weight(shape=[20 * 80 * 3, 26 * 4])
        bias = init_bias(shape=[26 * 4])

        image = tf.reshape(image, [-1, 20 * 80 * 3])

        return tf.matmul(tf.cast(image, tf.float32), weight) + bias



def codeRecog(file_path):
    """
    验证码识别
    :return:
    """
    image, label = read_and_decode(file_path=file_path)

    y_predict = fc_model(image)

    y_true = one_hot_encode(label)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.reshape(y_true, [10, 4 * 26]), logits=y_predict))

    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(
        tf.arg_max(tf.reshape(y_predict, [10, 4, 26]), 2), tf.arg_max(y_true, 2)
    ), tf.float32))

    init_op = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init_op)
        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(coord=coord, sess=session)

        for i in range(5000):
            session.run(train_op)
            print("第", (i + 1), "次训练准确率：", accuracy.eval())

        coord.request_stop()
        coord.join(threads=threads)



if __name__ == '__main__':
    codeRecog(file_path="D:\\develop\\PythonProjects\\MachineLearn\\tfrecords\\captcha.tfrecords")
