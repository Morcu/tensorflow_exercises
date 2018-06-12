import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def main():
    x_data, y_data = generate_points(1000)
    linear_regresion(x_data,y_data)


def generate_points(num_points):
    """Function that generates random points to make as an start point for the regresion"""
    points = []
    #Randomize numbers for the regresion
    for i in range(num_points):
        x1= np.random.normal(0.0, 0.55)
        y1= x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
        points.append([x1, y1])
    x_data = [v[0] for v in points]
    y_data = [v[1] for v in points]

    plt.plot(x_data, y_data, 'ro', label='Original data')
    plt.legend()
    plt.show() 
    return x_data, y_data
   
def linear_regresion(x_data, y_data):
    """Function that makes a linear regresion using tensorflow"""
    #Initialize variables
    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    b = tf.Variable(tf.zeros([1]))
    y = W * x_data + b
    #Cost function
    loss = tf.reduce_mean(tf.square(y - y_data))
    #GradientDescent
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    #Training Steps
    for step in range(10):
        sess.run(train)
        print(step, sess.run(W), sess.run(b))
        #Graphic display of the training
        plt.plot(x_data, y_data, 'ro', label='Iteration: '+ str(step))
        plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
        plt.xlabel('x')
        plt.xlim(-2,2)
        plt.ylim(0.1,0.6)
        plt.ylabel('y')
        plt.legend()
        plt.show() 

if __name__ == "__main__":
    main()