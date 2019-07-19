import tensorflow as tf

if __name__ == '__main__':
    a = [[1, 2, 3],
         [4, 5, 6]]
    b = [[7, 8, 9],
         [10, 11, 12]]
    c = tf.concat([a, b], axis=0)  # 按行合并a和b
    print(type(c))
    with tf.Session() as session:
        print("", c.eval())