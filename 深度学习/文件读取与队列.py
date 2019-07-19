import tensorflow as tf
import os

def asyncQueue():
    """
    tensorflow里的多线程与队列
    :return:
    """
    q = tf.FIFOQueue(1000, tf.float32)

    var = tf.Variable(1.0)
    data = tf.assign_add(var, tf.constant(1.0))  # 自增
    enqueue = q.enqueue(data)

    # 线程管理器，协调子线程的结束
    coord = tf.train.Coordinator()

    # 子线程执行队列操作，enqueue_ops决定创建多少个子线程
    qr = tf.train.QueueRunner(q, enqueue_ops=[enqueue] * 2)

    init_op = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init_op)
        threads = qr.create_threads(session, coord=coord, start=True)  # 创建线程，开始执行

        for i in range(300):  # 主线程读取队列
            print(session.run(q.dequeue()))

        # 管理器回收子线程
        coord.request_stop()
        coord.join(threads)



def readCsv(filelist):
    """
    读取csv文件列表
    :return: 读取的内容
    """
    # 构造字符串输入队列
    file_queue = tf.train.string_input_producer(filelist)
    # 文件读取器
    reader = tf.TextLineReader()

    # 读取文件，返回两个tensor
    key, value = reader.read(file_queue)

    # 解码文件
    records = [["None"], ["None"]]
    example, label = tf.decode_csv(value, record_defaults=records)

    # 批处理多个文件，batch_size决定总共读取多少数据，capacity表示队列容量，一般两者设为一样
    example_batch, label_batch = tf.train.batch([example, label], batch_size=9,
                                               num_threads=1, capacity=9)
    return example_batch, label_batch


def readpic(file_list):
    """
    读取图片列表
    :param file_list: 待读取的图片列表
    :return: 读取的结果张量
    """
    # 构造队列与阅读器
    file_queue = tf.train.string_input_producer(file_list)
    reader = tf.WholeFileReader()

    # 读取图片
    key, value = reader.read(file_queue)
    # 解码图片，统一大小为200 * 200，此处不可指定通道数
    image = tf.image.resize(tf.image.decode_jpeg(value), [200, 200])
    # 将内容转换成三维(加上通道数)
    image.set_shape([200, 200, 3])

    # 批处理
    image_batch = tf.train.batch([image], batch_size=20, num_threads=1, capacity=20)

    return image_batch


class CifarReader(object):
    def __init__(self, file_list):
        self.file_list = file_list

    def read_and_decode(self):
        file_queue = tf.train.string_input_producer(self.file_list)
        reader = tf.FixedLengthRecordReader(3073)

        key, value = reader.read(file_queue)

        image_data = tf.decode_raw(value, tf.uint8)

        label = tf.cast(tf.slice(image_data, [0], [1]), tf.int32)
        image = tf.slice(image_data, [1], [3072])

        image_finish = tf.reshape(image, [32, 32, 3])

        image_batch, label_batch = tf.train.batch([image_finish, label], batch_size=20,
                                                  num_threads=1, capacity=20)

        return image_batch, label_batch


    def write_tfRecords(self, image_batch, label_batch, dest_dir):
        """
        写入tfrecords文件
        :param image_batch: 待写入图像数据
        :param label_batch: 待写入标签数据
        :param dest_dir: 目标目录
        :return:
        """

        # 打开writer
        writer = tf.python_io.TFRecordWriter(dest_dir)

        # 写10个
        for i in range(10):
            image = image_batch[i].eval().tostring()
            label = (int)(label_batch[i].eval()[0])

            # 把每一组图像和标签解析成example张量
            # 图像对应bytes类型，label对应int64类型
            example = tf.train.Example(features=tf.train.Features(feature={
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            }))
            # 写入一个example
            writer.write(example.SerializeToString())

        writer.close()


    def read_tfRecord(self, file_list):
        """
        读取一组tfrecord文件
        :param file_list: 待读取文件列表
        :return: 图像和标签的batch张量
        """
        # 获取队列和reader
        file_queue = tf.train.string_input_producer(file_list)
        reader = tf.TFRecordReader()

        # 读取文件
        key, value = reader.read(file_queue)

        # 解析example里的features
        features = tf.parse_single_example(value, features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.uint64),

        })

        # 获取值并改变维度或转换类型
        image = tf.reshape(tf.decode_raw(features["image"], tf.uint8), [32, 32, 3])
        label = tf.cast(features["label"], tf.uint32)

        # 批处理
        return tf.train.batch([image, label], num_threads=1, batch_size=10, capacity=10)



if __name__ == '__main__':
    # asyncQueue()
    dir_path = "D:\\develop\\PythonProjects\\MachineLearn\\pic\\"
    # dir_path = "...\\bin"
    file_name_list = os.listdir(dir_path)
    # image_batch = readpic([os.path.join(dir_path, file) for file in file_name_list])
    image_batch, label = CifarReader(file_name_list).read_and_decode()
    with tf.Session() as session:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)
        # print(session.run([image_batch]))
        print(session.run([image_batch]))

        coord.request_stop()
        coord.join(threads)