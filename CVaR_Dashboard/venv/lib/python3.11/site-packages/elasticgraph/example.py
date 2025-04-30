# %% Libraries
# import networkx as nx
# import pandas as pd
# import numpy as np
# from elasticgraph import Elasticgraph

# %% RWS
# import pandas as pd
# from elasticgraph import Elasticgraph
# from d3graph import d3graph
# from d3graph import vec2adjmat, adjmat2vec

# df = pd.read_csv('./data/RWS_source_target.txt', sep=';', header=0, encoding='latin1')
# adjmat = vec2adjmat(source=df['source'].values, target=df['target'].values)

# # Initialize Elasticgraph
# d3 = Elasticgraph(single_click_expand=False)
# d3.graph(adjmat)
# d3.show(filepath='C://temp//RWS_elastic.html')

# # Initialize D3graph
# d3 = d3graph()
# d3.graph(adjmat)
# d3.show(filepath='C://temp//RWS_d3graph.html', figsize=(None, None))


# %% small example
from elasticgraph import Elasticgraph

# Initialize
d3 = Elasticgraph(radius=40, hull_offset=80)
# Load example
adjmat, _ = d3.import_example('karate')

# Process adjmat
d3.graph(adjmat)


# Make direct adjustments in the dictionary
# print(d3.D3graph.edge_properties)
# print(d3.D3graph.node_properties)

html = d3.show(filepath=None, notebook=False)
assert html is not None
html = d3.show(filepath=None, notebook=True)
assert html is None
html = d3.show(filepath='test.html', notebook=False)
assert html is None

# %%
from elasticgraph import Elasticgraph

# Initialize
d3 = Elasticgraph()

# Load example
adjmat, _ = d3.import_example('bigbang')

# Process adjmat
d3.graph(adjmat)
d3.set_node_properties(color='cluster', cmap='Set1')
d3.show()

# d3.set_edge_properties(directed=True, marker_end='', marker_color='#000000')
# d3.D3graph.edge_properties['Penny', 'Leonard']['marker_end']='arrow'
# d3.D3graph.edge_properties['Sheldon', 'Howard']['marker_end']='stub'
# d3.D3graph.edge_properties['Sheldon', 'Leonard']['marker_end']='circle'
# d3.D3graph.edge_properties['Rajesh', 'Penny']['marker_end']='square'
# d3.D3graph.edge_properties['Penny', 'Leonard']['marker_color']='#ff0000'
# d3.show(filepath='c:\\temp\\network2.html')

# d3.set_node_properties(color=adjmat.columns.values, size=[10, 20, 10, 10, 15, 10, 5])

# d3.D3graph.node_properties['Penny']['tooltip']='test\ntest2'
# d3.show(filepath='network3.html')
