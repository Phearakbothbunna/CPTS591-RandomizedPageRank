import networkx as nx
import random
import time
import matplotlib.pyplot as plt


# This calc the probability of hopping to each node based on number of incoming links
def hop_probability(graph):
    probabilities = {}
    for node in graph.nodes():
        # Calc the weight of the current node based on the number of inbound links
        weight = len(list(graph.predecessors(node)))
        # Store the weight of the current node
        probabilities[node] = weight

    # Calc the total weight of all nodes
    total_weight = sum(probabilities.values())
    for node in probabilities:
        # Normalize the weights to get probabilities (make sure that everything adds up to 1)
        probabilities[node] /= total_weight
    return probabilities

# This is our modified PageRank algo
def randomized_pagerank(graph, damping=0.8, num_iterations=10):
    # Initialize the random weights for each node (based on the number of inbound links)
    rand_weights = {}
    for node in graph.nodes():
        # Calculate the weight based on the number of inbound links
        inbound_links = len(list(graph.predecessors(node)))
        # Random weight within a range based on the number of inbound links
        # making sure that the weight is between 0.1 and 1
        weight = inbound_links * random.uniform(0.1, 1.0)
        rand_weights[node] = weight

    # We initialize the PageRank scores with equal probabilities for all nodes
    pagerank_score = {}
    for node in graph.nodes():
        pagerank_score[node] = 1 / len(graph)


    # Perform the random hops (we set the iterations to 10 for the randomized algorithm)
    # This simulates the behavior of web surfer who jumps to random pages
    # For each iteration, a random node is chosen (based on the weight assigned)
    for i in range(num_iterations):
        # Calculate the probabilities of hopping to each node
        probabilities = hop_probability(graph)
        # We choose a random node based on the probabilities calcualated above
        # [0] returns the 1st element, which is the node
        node = random.choices(list(graph.nodes()), weights=list(probabilities.values()))[0]
        # Then we can try increasing the PageRank score of the chosen node
        pagerank_score[node] += (1 - damping) / len(graph)

    # Perform PageRank iterations
    # This updates the pagerank score for each node (based on current score & link structure)
    for i in range(num_iterations):
        # New dict is created to store the updated pagerank scores
        new_pagerank = {}
        # This will iterate over each node in the graph we have
        for node in graph.nodes():
            pagerank_sum = 0
            # Calculate the sum of the pagerank scores of its incoming neighbors for each node
            for neighbor in graph.predecessors(node):
                edges = graph.get_edge_data(neighbor, node)
                if edges:
                    edge_weight_sum = sum(edges.values())
                # If there are no edges, we can just set it to 1 to make sure it doesn't give error in the calculation
                else:
                    edge_weight_sum = 1
                # This will find the importance of the neighbors and how it influences the current node's score
                neighbor_contribute = pagerank_score[neighbor] / edge_weight_sum
                pagerank_sum += neighbor_contribute
            # Calculate the new pagerank scores for the current node
            # (using the current pagerank score, damping & the sum from earlier)
            new_pagerank[node] = (1 - damping) + damping * pagerank_sum

        # Calculate the total pagerank score by summing all the new pagerank scores
        total_pagerank_score = sum(new_pagerank.values())
        # We divide each score by the total pagerank score above to normalize the score (make sure they add up to 1)
        for node, score in new_pagerank.items():
            pagerank_score[node] = score / total_pagerank_score
    return pagerank_score

def main():
    data_file = './web-Google.txt'
    # Read the Google web graph dataset and create a directed graph from it
    G = nx.read_adjlist(data_file, create_using=nx.DiGraph())
    sub_nodes = list(G.nodes())[:5000]
    # Generate the subgraph of the selected node
    subgraph = G.subgraph(sub_nodes)

    # We measure computation time for original PageRank algorithm
    start_time = time.time()
    pr_scores_original = nx.pagerank(subgraph)
    original_algo_time = time.time() - start_time

    # # Computation time for PageRank with random hops & random weights
    start_time = time.time()
    pr_scores_randomized = randomized_pagerank(subgraph)
    randomized_algo_time = time.time() - start_time

    # Print the computation time
    print(f"Computation time for Original PageRank: {original_algo_time} s")
    print(f"Computation time for Randomized PageRank: {randomized_algo_time} s")

    print("Original PageRank scores:")
    # Sort by PageRank scores in descending order (high to low)
    sorted_scores = sorted(pr_scores_original.items(), key=lambda item: item[1], reverse=True)
    for node, score in sorted_scores:
        print(f"Node: {node}, PageRank Score: {score}")
    print("\nPageRank scores with random hops & random weights:")
    # Sort by PageRank scores in descending order
    sorted_scores_rand = sorted(pr_scores_randomized.items(), key=lambda item: item[1], reverse=True)
    for node, score in sorted_scores_rand:
        print(f"Node: {node}, PageRank Score: {score}")


    # top_15_nodes = sub_nodes[:15]
    # visual_graph = G.subgraph(top_15_nodes)
    # # Visualization
    # nx.draw(visual_graph, with_labels=True)
    # plt.show()


main()