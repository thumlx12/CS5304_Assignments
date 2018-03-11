from assign1 import *


def visualize(cluster, data, target, legends):
    '''
    This function can take in data points that are from different clusters
    and present them in a graph
    :param cluster: the number of different clusters
    :param data: the input data (2 dimensional)
    :param target: the label of each entry in data
    :param legends:
    :return:
    '''
    centroids = []
    for k in range(cluster):
        indice = [i for i, j in enumerate(target) if j == k]
        points = data[indice]
        centroids.append(points.mean(axis=0))
        plt.scatter(points[:, 0], points[:, 1])

    for centroid in centroids:
        plt.scatter(centroid[0], centroid[1], marker='x')

    plt.legend(legends)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_labels", default="labels.txt")
    parser.add_argument("--path_to_ids", default="validation.txt")
    options = parser.parse_args()

    labels = load_labels(options.path_to_labels)
    train_data, train_target, _ = load_training_data()
    eval_data, eval_target, _ = load_validation_data(options.path_to_ids)

    # the following codes show how to find best K for each label in KNN
    # and output the max average accuracy for each label
    limit = 1000
    selected_labels = [4, 5, 7, 33, 59, 70, 83, 95, 98, 102]
    best_k_for_each_label = np.array(
        [findBestKNN(train_data[:limit], train_target[:limit], l) for l in selected_labels])
    print("\nBest K for each selected label is:\n", best_k_for_each_label[:, 1])

    # the following codes use BernoulliNB to predict the selected labels
    # and output the corresponding accuracy
    NB_ave_accuracy_each_label = np.array(
        [np.mean(cross_val_score(BernoulliNB(), train_data, train_target[:, l], cv=5)) for l in selected_labels])
    print("For each label, the accuracy would be", NB_ave_accuracy_each_label)

    # the following codes are for (1c). Data Analysis
    import matplotlib.pyplot as plt

    plt.plot(selected_labels, best_k_for_each_label[:, 0], marker='o')
    plt.plot(selected_labels, NB_ave_accuracy_each_label, marker='x')
    plt.legend(['KNN', 'BernoulliNB'])
    plt.show()

    # From the graph presented, one can see that label 33 is the most difficult label to classify

    # 2. KMeans

    kmeans_first_ten_label_classifiers = [CS5304KMeansClassifier() for i in range(len(selected_labels))]
    for i in range(len(selected_labels)):
        kmeans_first_ten_label_classifiers[i].train(train_data, train_target[:, selected_labels[i]])

    # 2b.
    prediction_result_on_label33 = np.array(kmeans_first_ten_label_classifiers[3].predict(train_data))
    train_data_dimension_reducted = TruncatedSVD(n_components=2).fit_transform(train_data

    # This line visualize the predictions from KMeans for label 33 with a 2-dimensional plot
    visualize(2, train_data_dimension_reducted, prediction_result_on_label33, ['p_l33_false', 'p_l33_true'])
    # This line visualize the original data distribution for label 33 with a 2-dimensional plot
    visualize(2, train_data_dimension_reducted, train_target[:, 33], ['l33_false', 'l33_true'])

    # 2c.
    KMeans_label_33_2cluster = KMeans(n_clusters=2, init='random').fit(train_data)
    KMeans_label_33_3cluster = KMeans(n_clusters=3, init='random').fit(train_data)
    KMeans_label_33_4cluster = KMeans(n_clusters=4, init='random').fit(train_data)

    two_cluster_prediction = KMeans_label_33_2cluster.predict(train_data)
    three_cluster_prediction = KMeans_label_33_3cluster.predict(train_data)
    four_cluster_prediction = KMeans_label_33_4cluster.predict(train_data)

    visualize(2, train_data_dimension_reducted, two_cluster_prediction, ['cluster 0', 'cluster 1'])
    visualize(3, train_data_dimension_reducted, three_cluster_prediction, ['cluster 0', 'cluster 1', 'cluster 2'])
    visualize(4, train_data_dimension_reducted, four_cluster_prediction,
              ['cluster 0', 'cluster 1', 'cluster 2', 'cluster 3'])
