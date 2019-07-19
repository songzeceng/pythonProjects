import tensorflow as tf

if __name__ == '__main__':
    plt = tf.placeholder(tf.float32, [None, 3])
    plt2 = tf.placeholder(tf.float32, [None, 2])
    print(plt2)
    plt2.set_shape([4, 2])  # 形状固定，不可再次修改
    print(plt2)
    plt3 = tf.reshape(plt2, [2, 4])  # 可通过reshape()创造新的张量对象，看要注意元素数量不能变，不能跨维度
    print(plt3)
    # 二维浮点张量占位符

    var = tf.Variable(tf.random_normal([2, 3], mean=0, stddev=1.0))  # 变量，可持久化保存的张量
    a = tf.constant([1, 2, 3, 4, 5])
    init_op = tf.global_variables_initializer()
    print(var)

    b = tf.constant(3.2)
    c = tf.constant(2.1)
    d = tf.add(b, c)
    with tf.Session() as session:
        session.run(init_op)  # 变量必须初始化
        print("", session.run(plt, feed_dict={
            plt: [[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]]
        }))
        print("plt.shape:", plt.shape)
        print(session.run([a, var]))
        print(session.run(d))
        tf.summary.FileWriter("D:\\develop\\PythonProjects\\MachineLearn",
                              graph=session.graph)