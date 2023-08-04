from flask import Flask, request, jsonify
from utils.algorithm import *
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

@app.route('/cluster', methods=['POST'])
def cluster_data():
    # Get the JSON data from the client
    data = request.json["data"]  

    # Convert the data to a numpy array
    num_data_points = len(data)
    data_points = np.zeros((num_data_points, 6))
    for i, d in enumerate(data):
        data_points[i] = [d["wave"], d["wind"], d["current"], d["salinity"], d["temperature"], d["catchResult"]]

    print(data_points)
    # Perform the Fuzzy C-Means clustering on the data
    num_clusters = 2  # You can set the number of clusters here
    fuzziness = 2
    max_iterations = 100
    tolerance = 1e-6

    cluster_centers, membership_matrix = fuzzy_c_means(data_points, num_clusters, fuzziness, max_iterations, tolerance)

    # Get the clusters by choosing the index of the maximum value in the membership matrix
    clusters = np.argmax(membership_matrix, axis=1)

    # Prepare the response data
    response_data = []
    for i, d in enumerate(data):
        d["cluster"] = int(clusters[i])  # Convert to integer to ensure JSON serialization
        response_data.append(d)

    # Calculate the Fuzzy Partition Coefficient (FPC)
    fpc = fuzzy_partition_coefficient(membership_matrix)

    # Prepare the cluster center data
    cluster_centers_list = cluster_centers.tolist()

    # Return the FPC and cluster centers in the response
    response = {
        "fpc": fpc,
        "cluster_centers": cluster_centers_list,
        "data": response_data
    }

    return jsonify(response)

@app.route('/', methods=["GET"])
def index():
    return "Hello from Flask"

if __name__ == '__main__':
    app.run(debug=True)
