import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import msvcrt
import networkx as nx
import os


#####################
# Utility functions #
#####################

def save_figure(folder, filename, fig):
    """Save figure to disk"""
    fn = os.path.join(folder, filename)
    fig.savefig(fn, dpi=fig.dpi, bbox_inches='tight', pad_inches=0.5)


######################
# Helpers for graphs #
######################

def create_target(nodes, edges, name):
    """"Return the graph corresponding to the optimal design.
    
    Parameters
    ----------
    nodes : list
        Nodes of the target graph.
    layer : list of tuples
        Edges of the target graph.
    name : str
        Name of the target graph.

    Returns
    -------
    target : `networkx.Graph`
        Graph representing the optimal design.
    """
    target = nx.Graph(name=name)
    target.add_nodes_from(nodes)
    target.add_edges_from(edges)
    return target


def get_essential_nodes(target):
    """"Return the nodes that are linked through an edge in the optimal graph.
    
    Parameters
    ----------
    target : `networkx.Graph`
        Graph representing the optimal design.

    Returns
    -------
    essential : set
        Nodes that are linked by edges in the target graph.
    """
    essential = set()
    for start, end in target.edges:
        essential.add(start)
        essential.add(end)
    return essential


def get_required_nodes(target):
    """"Return the nodes in the optimal graph that are isolated.
    
    Parameters
    ----------
    target : `networkx.Graph`
        Graph representing the optimal design.

    Returns
    -------
    required : set
        Nodes the isolated nodes of the target graph (those nodes that are 
        not linked by any edge in the optimal graph).
    """
    essential = get_essential_nodes(target)
    required = set(target.nodes).difference(essential)
    return required


#########################################
# Wrappers for easy creation of figures #
#########################################

def node_colors(graph, essential, required,
                color_essential='lightgreen', 
                color_required='skyblue', 
                color_other='orange'):
    """"Map nodes to colors.
    
    Parameters
    ----------
    graph : `networkx.Graph`
        Graph that represents a layout.
    essential : set
        Nodes that are linked by edges in the target graph.
    required : set
        Isolated nodes in the target graph.
    color_essential : string
        Color for the nodes which are lnked through edges in the 
        target graph.
    color_required : string
        Color for the isolated nodes of the target graph.
    color_other : string
        Color for the nodes which are not in the target graph.

    Returns
    -------
    cmap : list
        Color map for the different node types.
    """
    cmap = []
    for node in graph.nodes:
        if int(node) in essential:
            cmap.append(mcolors.CSS4_COLORS[color_essential])
        elif int(node) in required: 
            cmap.append(mcolors.CSS4_COLORS[color_required])
        else:
            cmap.append(mcolors.CSS4_COLORS[color_other])
    return cmap


def draw_graph(graph, ax, cmap, title, node_size=1400, font_size=18):
    """"Wrapper for `nx.draw_circular`."""
    nx.draw_circular(graph, with_labels=True, ax=ax, font_size=font_size, 
                     node_size=node_size, node_color=cmap)
    ax.set_xlim([1.1*x for x in ax.get_xlim()])
    ax.set_ylim([1.1*y for y in ax.get_ylim()])
    ax.set_title(title, fontsize=font_size)


def draw_legend(graph, key, ax, props, font_size=14):
    """Add legend to graph plot."""
    items = [f'{num} - {name}' 
             for num, name in key.items() 
             if num in graph.nodes]
    txt = '\n'.join(items)
    ax.text(0, 1, txt, transform=ax.transAxes,
            fontsize=font_size, verticalalignment="top", bbox=props)
    ax.set_axis_off()


def draw_target(folder, target, essential, required, key, 
                props, show_legend=False, size=6):
    """Draw target graph."""
    if show_legend:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(2*size, size))
    else:
        fig, ax0 = plt.subplots(1, 1, figsize=(size, size))

    cmap = node_colors(target, essential, required)
    draw_graph(target, ax0, cmap, f'{target.name} graph')
    if show_legend:
        draw_legend(target, key, ax1, props)

    plt.show(fig)
    filename = f'graph-{target.name.lower()}.png'
    save_figure(folder, filename, fig)


def draw_solution(folder, full, pruned, essential, required, key, 
                  props, show_legend=False, size=6):
    """"Draw graphs (complete and simplified) of a proposed solution."""
    number = full.name
    if show_legend:
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(3*size, size))
    else:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(2*size, size))

    cmap_full = node_colors(full, essential, required)
    cmap_pruned = node_colors(pruned, essential, required)

    draw_graph(full, ax0, cmap_full, 'Complete')
    draw_graph(pruned, ax1, cmap_pruned, 'Simplified')

    if show_legend:
        draw_legend(full, key, ax2, props)

    fig.suptitle(f'Graph {number}', fontsize=18)
    plt.show(fig)
    filename = f'graph-{number}.png'
    save_figure(folder, filename, fig)