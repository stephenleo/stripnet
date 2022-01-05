import logging
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import plotly.graph_objects as go
import plotly.express as px
from typing import Tuple
from . import utils
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger('main')


class StripNet:
    def __init__(self, model='allenai-specter'):
        self.model = SentenceTransformer(model)

        self.centrality_mapping = {
            'Closeness Centrality': nx.closeness_centrality,
            'Degree Centrality': nx.degree_centrality,
            'Eigenvector Centrality': nx.eigenvector_centrality,
            'Betweenness Centrality': nx.betweenness_centrality,
        }

    def load_bertopic_model(self, min_topic_size: int = 10, n_gram_range: tuple = (1, 1), stop_words: str = 'english', verbose: bool = True) -> BERTopic:
        '''load_bertopic_model Initialize a BERTopic model with the provided arguments

        Args:
            min_topic_size (int, optional): [description]. Defaults to 10.
            n_gram_range (tuple, optional): [description]. Defaults to (1, 1).
            stop_words (str, optional): [description]. Defaults to 'english'.
            verbose (bool, optional): [description]. Defaults to True.

        Returns:
            BERTopic: Initialized BERTopic model
        '''
        return BERTopic(
            vectorizer_model=CountVectorizer(
                stop_words=stop_words, ngram_range=n_gram_range
            ),
            min_topic_size=min_topic_size,
            verbose=verbose,
        )

    def embedding_gen(self, text: pd.Series) -> np.ndarray:
        '''embedding_gen Generate the embeddings for the `text` column

        Args:
            text (pd.Series): Pandas dataframe column containing the text on which to run the STriPNet pipeline

        Returns:
            np.ndarray: embeddings of the text
        '''
        return self.model.encode(text)

    def topic_modeling(self, text: pd.Series, min_topic_size: int = 10, n_gram_range: tuple = (1, 1), stop_words: str = 'english', verbose: bool = True) -> Tuple[pd.DataFrame, dict]:
        '''topic_modeling Run the BERTopic modeling on the corpus

        Args:
            text (pd.Series): Pandas dataframe column containing the text on which to run the STriPNet pipeline
            min_topic_size (int, optional): [description]. Defaults to 10.
            n_gram_range (tuple, optional): [description]. Defaults to (1, 1).
            stop_words (str, optional): [description]. Defaults to 'english'.
            verbose (bool, optional): [description]. Defaults to True.

        Returns:
            pd.DataFrame: Topics datafame including the topics for each row of input text
            dict: Topics dictionary of the form {Topic: "Topic_Name"}
        '''
        topic_data = pd.DataFrame(text)
        topic_data.columns = ['Text']
        topic_data['Text'] = topic_data['Text'].apply(
            utils.text_processing)

        logger.info('Initializing the topic model')
        self.bertopic_model = self.load_bertopic_model(
            min_topic_size, n_gram_range, stop_words=stop_words, verbose=verbose)

        logger.info('Training the topic model')
        topic_data["Topic"], topic_data["Probs"] = self.bertopic_model.fit_transform(
            text, embeddings=self.embeddings)

        logger.info('Populating Topic Results')
        topic_df = self.bertopic_model.get_topic_info()
        topic_df.columns = ['Topic', 'Topic_Count', 'Topic_Name']
        topic_df['Topic_Name'] = topic_df['Topic_Name'].apply(
            lambda x: ', '.join(x.split('_')[1:]))
        topic_df = topic_df.sort_values(by='Topic_Count', ascending=False)
        topic_data = topic_data.merge(topic_df, on='Topic', how='left')

        topics = topic_df.head(10).set_index('Topic').to_dict(orient='index')

        return topic_data, topics

    def stripnet(self, topic_data: pd.DataFrame, topics: dict, threshold: float = None, max_connections: int = None, remove_isolated_nodes: bool = False) -> Tuple[nx.Graph, Network]:
        '''stripnet Generate the STriPNet

        Args:
            topic_data (pd.DataFrame): Topics datafame including the topics for each row of input text
            topics (dict): Topics dictionary of the form {Topic: "Topic_Name"}
            threshold (float, optional): Minimum cosine similarity to draw a link on the network. Default value None will use an internally calculated threshold value.
            max_connections (int, optional): Maximum connections to allow in the network. The actual value used might be lower than this due to internal calculations. Defaults to None which uses the internally generated heuristic for max_connections
            remove_isolated_nodes (bool, optional): [description]. Defaults to False.
        Returns:
            nx.Graph: NetworkX graph of the STriPNet
            Network: Pyvis graph of the STriPNet for plotting
        '''
        cosine_sim_matrix = utils.cosine_sim(self.embeddings)

        if threshold:
            self.threshold = threshold
        else:
            self.threshold, min_value = utils.calc_optimal_threshold(
                cosine_sim_matrix,
                # 25% is a good value for the number of papers
                max_connections=min(
                    utils.calc_max_connections(
                        len(self.text), 0.25), self.max_connections
                )
            )

        self.neighbors, num_connections = utils.calc_neighbors(
            cosine_sim_matrix, threshold=self.threshold)
        logger.info(f'Number of connections: {num_connections}')

        nx_net, pyvis_net = utils.network_plot(
            topic_data, topics, self.neighbors, remove_isolated_nodes)

        return nx_net, pyvis_net

    def most_important(self, centrality_option: str = 'Betweenness Centrality') -> go.Figure:
        '''most_important Plot most important texts as per network centrality calculation

        Args:
            centrality_option (str, optional): The network centrality measure to use. Defaults to 'Betweenness Centrality'.

        Returns:
            go.Figure: Plotly graph object plot
        '''

        logger.info('Calculating Network Centrality')
        centrality = self.centrality_mapping[centrality_option](self.nx_net)

        # Sort Top 10 Central nodes
        central_nodes = sorted(
            centrality.items(), key=lambda item: item[1], reverse=True)
        central_nodes = pd.DataFrame(central_nodes, columns=[
            'node', centrality_option]).set_index('node')

        joined_data = self.topic_data.join(central_nodes)

        top_central_nodes = joined_data.sort_values(
            centrality_option, ascending=False).head(10)

        # Prepare for plot
        top_central_nodes = top_central_nodes.reset_index()
        top_central_nodes['index'] = top_central_nodes['index'].astype(str)

        # Plot the Top 10 Central nodes
        fig = px.bar(top_central_nodes, x=centrality_option, y='index',
                     color='Topic_Name', hover_data=['Text'], orientation='h')
        fig.update_layout(yaxis={'categoryorder': 'total ascending', 'visible': False, 'showticklabels': False},
                          font={'size': 15}, height=800)
        return fig

    def fit_transform(self,
                      text: pd.Series,
                      min_topic_size: int = 10,
                      n_gram_range: tuple = (1, 1),
                      stop_words: str = 'english',
                      threshold: float = None,
                      remove_isolated_nodes: bool = False,
                      max_connections: int = None,
                      verbose: bool = True) -> None:
        '''fit_transform Run the STriPNet modeling pipeline.

        [extended_summary]

        Args:
            text (pd.Series): Pandas dataframe column containing the text on which to run the STriPNet pipeline.
            min_topic_size (int, optional): [description]. Defaults to 10.
            n_gram_range (tuple, optional): [description]. Defaults to (1, 1).
            stop_words (str, optional): [description]. Defaults to 'english'.
            threshold (float, optional): Minimum cosine similarity to draw a link on the network. Default value None will use an internally calculated threshold value.
            remove_isolated_nodes (bool, optional): [description]. Defaults to False.
            max_connections (int, optional): Maximum connections to allow in the network. The actual value used might be lower than this due to internal calculations. Defaults to None which uses the internally generated heuristic for max_connections
            verbose (bool, optional): [description]. Defaults to True.
        '''
        if text.isna().sum() > 0:
            logger.info('Missing data detected. Dropping them')
            text = text.dropna().reset_index(drop=True)

        self.text = text
        self.remove_isolated_nodes = remove_isolated_nodes

        if max_connections:
            self.max_connections = max_connections
        else:
            self.max_connections = utils.calc_max_connections(
                len(self.text), 1)

        logger.info('========== Step1: Calculating Embeddings ==========')
        self.embeddings = self.embedding_gen(self.text)

        logger.info('========== Step2: Topic modeling ==========')
        self.topic_data, self.topics = self.topic_modeling(self.text, min_topic_size=min_topic_size,
                                                           n_gram_range=n_gram_range, stop_words=stop_words,
                                                           verbose=verbose)

        logger.info('========== Step3: STriP Network ==========')
        self.nx_net, self.pyvis_net = self.stripnet(
            self.topic_data, self.topics, threshold=threshold, remove_isolated_nodes=self.remove_isolated_nodes, max_connections=self.max_connections)

        logger.info('========== Model Fit Successfully! ==========')
        self.pyvis_net.show('stripnet.html')
