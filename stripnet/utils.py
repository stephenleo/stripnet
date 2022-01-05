import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple
import textwrap
import logging
logger = logging.getLogger('main')


def cosine_sim(embeddings):
    logger.info('Cosine similarity')
    cosine_sim_matrix = cosine_similarity(embeddings)

    # Take only upper triangular matrix
    cosine_sim_matrix = np.triu(cosine_sim_matrix, k=1)

    return cosine_sim_matrix


def calc_max_connections(num_rows, ratio):
    n = ratio*num_rows

    return n*(n-1)/2


def calc_neighbors(cosine_sim_matrix, threshold):
    neighbors = np.argwhere(cosine_sim_matrix >= threshold).tolist()

    return neighbors, len(neighbors)


def calc_optimal_threshold(cosine_sim_matrix, max_connections):
    """Calculates the optimal threshold for the cosine similarity matrix.
    Allows a max of max_connections
    """
    logger.info('Calculating optimal threshold')
    thresh_sweep = np.arange(0.05, 1.05, 0.05)[::-1]
    for idx, threshold in enumerate(thresh_sweep):
        _, num_neighbors = calc_neighbors(
            cosine_sim_matrix, threshold)
        if num_neighbors > max_connections:
            break

    return round(thresh_sweep[idx-1], 2).item(), round(thresh_sweep[idx], 2).item()


def text_processing(text):
    text = text.split('[SEP]')
    text = '<br><br>'.join(text)
    text = '<br>'.join(textwrap.wrap(text, width=50))[:500]
    text = text + '...'
    return text


def network_plot(topic_data: pd.DataFrame,
                 topics: dict,
                 neighbors: np.ndarray,
                 remove_isolated_nodes: bool = False) -> Tuple[nx.Graph, Network]:
    '''network_plot Creates a network plot of the connected texts. Colored by Topic Model topics.

    Args:
        topic_data (pd.DataFrame): Topics datafame including the topics for each row of input text
        topics (dict): Topics dictionary of the form {Topic: "Topic_Name"}
        neighbors (np.ndarray): Connected nodes in the Graph
        remove_isolated_nodes (bool, optional): Defaults to False. Setting this to True removes nodes that are not connected to any other nodes from the visualization

    Returns:
        nx.Graph: NetworkX graph of the STriPNet
        Network: Pyvis graph of the STriPNet for plotting
    '''

    logger.info('Calculating Network Plot')
    nx_net = nx.Graph()
    pyvis_net = Network(height='750px', width='100%', bgcolor='#222222')

    # Add Nodes
    nodes = [
        (
            row.Index,
            {
                'group': row.Topic,
                'label': row.Index,
                'title': row.Text,
                'size': 20, 'font': {'size': 20, 'color': 'white'}
            }
        )
        for row in topic_data.itertuples()
    ]
    nx_net.add_nodes_from(nodes)
    assert(nx_net.number_of_nodes() == len(topic_data))

    # Add Edges
    nx_net.add_edges_from(neighbors)
    assert(nx_net.number_of_edges() == len(neighbors))

    # Optimization: Remove Isolated nodes
    if remove_isolated_nodes:
        nx_net.remove_nodes_from(list(nx.isolates(nx_net)))

    # Add Legend Nodes
    step = 150
    x = -2000
    y = -500
    legend_nodes = [
        (
            len(topic_data)+idx,
            {
                'group': key, 'label': value['Topic_Name'],
                'size': 30, 'physics': False, 'x': x, 'y': f'{y + idx*step}px',
                # , 'fixed': True,
                'shape': 'box', 'widthConstraint': 1000, 'font': {'size': 40, 'color': 'black'}
            }
        )
        for idx, (key, value) in enumerate(topics.items())
    ]
    nx_net.add_nodes_from(legend_nodes)

    # Plot the Pyvis graph
    pyvis_net.from_nx(nx_net)

    return nx_net, pyvis_net
