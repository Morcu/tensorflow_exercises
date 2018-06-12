import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf

def main():
    """
        The clustering method is K-means
    """
    points = generate_points(2000)
    clustering_k_means(points, 4, 100)


def generate_points(num_points):
    """
        Function that generates random points to make as an start point for the clustering
    """
    
    points = []
    for i in range(num_points):
        if (np.random.random() > 0.5):
            points.append([np.random.normal(0.0, 0.9), np.random.normal(0.0, 0.9)])
        else:
            points.append([np.random.normal(3.0, 0.5), np.random.normal(1.0, 0.5)]) 

    #Plot in a 2D graphic
    df = pd.DataFrame({"x": [v[0] for v in points], "y": [v[1] for v in points]})
    sns.lmplot("x", "y", data=df, fit_reg=False, size=6)
    plt.show() 
    return points

def clustering_k_means(points, k, num_steps):
    """
        Function that makes a linear regresion using tensorflow
    """

    vectors = tf.constant(points)

    #Choose random the centroids using a given K
    centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors), 
                                    [0,0],[k,-1]))
    
    expanded_vectors = tf.expand_dims(vectors, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)

    print(expanded_vectors.get_shape())
    print(expanded_centroids.get_shape())

    #K-means
    distances = tf.reduce_sum(
    tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)
    assignments = tf.argmin(distances, 0)

    means = tf.concat([
    tf.reduce_mean(
        tf.gather(vectors, 
                    tf.reshape(
                    tf.where(
                        tf.equal(assignments, c)
                    ),[1,-1])
                ),reduction_indices=[1])
    for c in range(k)], 0)

    update_centroids = tf.assign(centroids, means)
    init_op = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init_op)
    
    for step in range(num_steps):
        _, centroid_values, assignment_values = sess.run([update_centroids,
                                                        centroids,
                                                        assignments])
    print ("centroids")
    print (centroid_values)

    #Plot the final values
    data = {"x": [], "y": [], "cluster": []}
    for i in range(len(assignment_values)):
        data["x"].append(points[i][0])
        data["y"].append(points[i][1])
        data["cluster"].append(assignment_values[i])
    df = pd.DataFrame(data)
    sns.lmplot("x", "y", data=df, 
            fit_reg=False, size=7, 
            hue="cluster", legend=False)
    plt.show()

if __name__ == "__main__":
    main()