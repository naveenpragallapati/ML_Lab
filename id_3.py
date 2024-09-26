import pandas as pd
import numpy as np
from graphviz import Digraph


def visualize_tree(tree, features, parent_name='', graph=None, is_root=True):
    if graph is None:
        graph = Digraph()
        graph.node(name='root', label=None, shape='ellipse')
        parent_name = 'root'

    if isinstance(tree, dict):
        for k, v in tree.items():
            if isinstance(v, dict):
                node_name = f'{k}'
                if is_root:
                    is_root = False
                else:
                    if node_name not in features:
                        graph.node(name=node_name, label=str(k), shape='box')
                    else:
                        graph.node(name=node_name, label=str(k), shape='ellipse')
                graph.edge(parent_name, node_name)
                visualize_tree(v, features, node_name, graph, is_root)
            else:
                if v == 'Yes' or v == 'No':
                    node_name = f'{k}'
                    if node_name not in features:
                        graph.node(name=node_name, label=str(k), shape='box')
                    else:
                        graph.node(name=node_name, label=str(k), shape='ellipse')
                    graph.edge(parent_name, node_name)
                    node_name = f'{k}_{v}'
                    graph.node(name=node_name, label=str(v), shape='diamond')
                    graph.edge(str(k), node_name)
    else:
        if tree == 'Yes' or tree == 'No':
            node_name = f'{parent_name}_{tree}'
            graph.node(name=node_name, label=str(tree), shape='diamond')
            graph.edge(parent_name, node_name)

    return graph

def entropy(target_col):
    """Calculate the entropy of a dataset."""
    elements, counts = np.unique(target_col, return_counts=True)
    entropy_value = np.sum([(-counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy_value

def info_gain(data, split_attribute_name, target_name="class"):
    """Calculate the entropy of the total dataset"""
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    '''Calculate the weighted entropy'''
    weighted_entropy = np.sum([(counts[i]/np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name]) for i in range(len(vals))])
    '''Calculate the information gain'''
    information_gain = total_entropy - weighted_entropy
    return information_gain

def id3(data, original_data, features, target_attribute_name="class", parent_node_class=None):
    """Define the stopping criteria --> If one of this is satisfied, we want to return a leaf node"""
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    elif len(data) == 0:
        return np.unique(original_data[target_attribute_name])[np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])]
    elif len(features) == 0:
        return parent_node_class
    else:
        '''Define the parent node '''
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        '''Select the feature which best splits the dataset'''
        item_values = [info_gain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        '''Create the tree structure. The root gets the name of the feature (best_feature) with the maximum information gain in the first run'''
        tree = {best_feature: {}}
        features = [i for i in features if i != best_feature]
        '''Grow a branch under the root node for each possible value of the root node feature'''
        for value in np.unique(data[best_feature]):
            sub_data = data.where(data[best_feature] == value).dropna()
            '''Call the ID3 algorithm for each of those sub_datasets with the new parameters --> Here the recursion comes in! '''
            subtree = id3(sub_data, original_data, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree
        return tree

# Training Examples:
training_data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
})
feature_columns = training_data.columns[:-1].to_list()
target_column = 'PlayTennis'
decision_tree = id3(training_data, training_data, feature_columns, target_column)
print("Decision Tree:",decision_tree)
decision_tree_graph = visualize_tree(decision_tree, feature_columns)
decision_tree_graph.render('decision_tree', format='png', cleanup=True)
