import tensorflow as tf

if __name__ == '__main__':
    graph = tf.get_default_graph()
    # 默认的图，相当于给程序分配内存
    print(graph)

    # 构建新的上下文图，包括一组op(api函数)和tensor(数据)
    g = tf.Graph()
    with g.as_default():
        c = tf.constant(7.0)
        print(c.graph)

    a = tf.constant(5.0)
    b = tf.constant(6.0)
    c = 3.2
    print(a + c)
    print("a=", a)
    print("b=", b)

    sum1 = tf.add(a, b)
    print("sum1=", sum1)
    with tf.Session(graph=graph) as session:
        # 会话，运行图的结构、计算，掌管和分配资源
        # 只能运行一张图
        print(session.run(sum1))
        print(a.graph)

        session.close()