import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("job_name", " ", "启动服务的类型")
tf.app.flags.DEFINE_integer("task_index", 0, "任务索引")


def main(argv):
    # 定义全局计数op，供hook使用
    global_step = tf.contrib.framework.get_or_create_global_step()

    # 指定集群描述对象，参数服务器和worker
    cluster = tf.train.ClusterSpec({
        "ps": [""],
        "worker": ["192.168.0.102:2222"]
    })

    # 创建不同的服务
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    # 不同的服务做不同的事情
    if FLAGS.job_name == "ps":
        # 参数服务器等待worker传参
        server.join()
    elif FLAGS.job_name == "worker":
        # worker可以指定设备运行
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:0/cpu:0/",
                cluster=cluster
        )):
            x = tf.Variable([1, 2, 3, 4])
            w = tf.Variable([1], [2], [3], [4])

            mat = tf.matmul(x, w)

        # 分布式会话
        with tf.train.MonitoredTrainingSession(
            master="grpc://192.168.0.102:2222",  # 指定主worker
            is_chief=FLAGS.task_index == 0,  # 是否主worker
            config=tf.ConfigProto(log_device_placement=True),
            hooks=tf.train.StopAtStepHook(last_step=500)
        ) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(mat)


if __name__ == '__main__':
    tf.app.run()