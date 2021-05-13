import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import write_dot
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.patches as mpatches
from mpl_toolkits.basemap import Basemap
import warnings
warnings.filterwarnings('ignore')


def extractTree(ns_output_dir):
    """
    Extract the complete hierarchical structure tree from NetStruct output files. 
    
    Parameters
    ----------
    netstruct_output_path : str
        Directory containing NetStruct output files
    
    Returns
    -------
    belonging_map : dict
        Map with node as key and list of individual indices clustered to the node as value
    parent_map : dict
        Map with node as key and its parent node as value
    child_map : dict
        Map with node as key and list of its child nodes as value
    node_list: list
        List of all nodes in the tree
    G_whole : networkx graph
        The complete hierarchical structure tree
    """
    
    
    files = [f for f in os.listdir(ns_output_dir) if re.match(r'Le_[0-9]+_En_[0-9]+_PaLe_[0-9]+_PaEn_[0-9]+_PaLi_[0-9]+_TH_.*_C.*', f)]

    fileInfo = list()
    file0 = [f for f in os.listdir(ns_output_dir) if re.match(r'Le_0_En_0_TH_*',f)][0]
    file0_info = re.findall(r'\d+', file0)
    fileInfo.append( (file0_info[0:2] + [None, None, None] + file0_info[2:]) )

    for f in files:
        fileInfo.append(re.findall(r'\d+', f))

    files.insert(0, file0)
    fileInfo = sorted(fileInfo, key=lambda x: int(x[0]))

    belonging_map = dict()
    parent_map = dict()
    child_map = dict()

    for fileIdx in range(len(fileInfo)):
        info = fileInfo[fileIdx]
        le = info[0]
        en = info[1]
        if le=='0':
            nodeFile = "Le_0_En_0_TH_0.0000000_C.txt"
        else:
            nodeFile = "Le_{}_En_{}_PaLe_{}_PaEn_{}_PaLi_{}_TH_{}.{}_C.txt".format(le,en,info[2],info[3],info[4],info[5],info[6])

        f1 = open(ns_output_dir+nodeFile, 'r')
        lines = f1.readlines()
        f1.close()

        s_raw = [l.split() for l in lines]
        s_all = [list([int(i) for i in subs]) for subs in s_raw]

        l_comms = list()
        for comm_idx in range(len(s_all)):
            comm = s_all[comm_idx]
            clusterID = '-'.join([le,en,str(comm_idx)])
            belonging_map[clusterID] = comm
            if fileIdx == 0:
                parent_map[clusterID] = 'ROOT'
                if 'ROOT' not in child_map:
                    child_map['ROOT'] = [clusterID]
                else:
                    child_map['ROOT'].append(clusterID)
            else:
                par = '-'.join(info[2:5])
                parent_map[clusterID] = par
                if par not in child_map:
                    child_map[par] = [clusterID]
                else:
                    child_map[par].append(clusterID)

    # whole tree
    root_whole = 'ROOT'

    G_whole = nx.DiGraph()
    below_root = child_map[root_whole].copy()
    G_whole.add_node(root_whole)

    for i,c in enumerate(below_root):
        G_whole.add_node(c)
        G_whole.add_edge(root_whole, c)

    while len(below_root) != 0:
        nextCluster = below_root.pop()
        if nextCluster in child_map:
            children = child_map[nextCluster]
            G_whole.add_node(nextCluster)

            for i,c in enumerate(children):
                G_whole.add_node(c)
                G_whole.add_edge(nextCluster, c)

            below_root.extend(children)

    node_list = list(G_whole.nodes)
    
    return belonging_map, parent_map, child_map, node_list, G_whole


def plotSubTree(child_map, belonging_map, cmap, **kwargs):
    """
    Plot the (sub)tree with gradient coloring scheme. 
    
    Parameters
    ----------
    child_map : dict
        Map with node as key and list of its child nodes as value
    belonging_map : dict
        Map with node as key and list of individual indices clustered to the node as value
    cmap : ColorMap
        Colormap used for the gradient coloring scheme 
    root : str, optional
        ID for the root of the (sub)tree with 'x-x-x' format (default is '0-0-0')
    tree_aspect : float, optional
        Aspect ratio for tree plot (default is 1.0)
    label_node : boolean, optional
        Whether to label each node on the side (default is True)
    node_size : int, optional
        Size of nodes in the tree plot (default is 0)
        Set to fixed size if given a non-zero value 
        Sset to variable set proportional to cluster size scaled by node_size_aspect if given 0 
    node_size_aspect : int, optional
        Scale factor of variable node size (default is 5)
        Node size is set to number of individuals in cluster multiplied by the scale factor if node_size = 0
    [Other optional arguments for plt.figure() function]:
        E.g. figsize=(5,5),dpi=100
    
    
    Returns
    -------
    fig
        Figure handle to the tree plot
    G : networkx graph
        Subtree plotted
    color_int_map : dict
        Map with node as key and its color range denoted by [lb,ub] (a subset of [0,1]) as value
    """
    
    # default arguments
    
    root = kwargs.pop('root', '0-0-0')
    tree_aspect = kwargs.pop('tree_aspect', 1.0)
    label_node = kwargs.pop('label_node', True)
    node_size = kwargs.pop('node_size', 0)
    node_size_aspect = kwargs.pop('node_size_aspect', 5)
    save_dir = kwargs.pop('save_dir', None)
    
    
    # extract subtree
    
    internal = [root]
    color_int_map = dict()
    color_int_map[root] = [0,1]
    G = nx.DiGraph()
    below_root = child_map[root].copy()
    G.add_node(root)
    subn = len(below_root)
    ca = color_int_map[root][0]
    cb = color_int_map[root][1]

    for i,c in enumerate(below_root):
        G.add_node(c)
        G.add_edge(root, c)
        color_int_map[c] = [ca+(cb-ca)*(i)/subn, ca+(cb-ca)*(i+1)/subn]

    while len(below_root) != 0:
        nextCluster = below_root.pop()

        if nextCluster in child_map:
            internal.append(nextCluster)
            children = child_map[nextCluster]
            G.add_node(nextCluster)
            subn = len(children)
            ca = color_int_map[nextCluster][0]
            cb = color_int_map[nextCluster][1]

            for i,c in enumerate(children):
                G.add_node(c)
                G.add_edge(nextCluster, c)
                color_int_map[c] = [ca+(cb-ca)*(i)/subn, ca+(cb-ca)*(i+1)/subn]

            below_root.extend(children)

    nodes = list(G.nodes)
    
    # plot subtree
    
    fig=plt.figure(**kwargs)
    plt.box(False)
    plt.axis('off')
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')

    pos =graphviz_layout(G, prog='dot')
    ax=plt.axes([0,0,1,1])
    ax.set_aspect(tree_aspect)
    nx.draw_networkx_edges(G,pos,ax=ax)
    nx.draw(G, pos, ax=ax, with_labels=False, arrows=False, node_size=1, alpha=0)

    # plot colored nodes
    
    for node in G.nodes():
        x,y=pos[node]
        col = cmap((color_int_map[node][0]+color_int_map[node][1])/2)
        if node_size:
            ax.scatter(x, y, c=[col], s=node_size)
        else:
            ax.scatter(x, y, c=[col], s=len(belonging_map[node])*node_size_aspect)
        if label_node:
            ax.text(x, y, node, c = (0,0,0,0.9), fontsize=8)
    
    return fig, G, color_int_map
             

def plotMap(df_meta, belonging_map, color_int_map, G, map_bdry, cmap, **kwargs):
    """
    Plot individuals on the map based on their clustering results,
    colored by the same scheme as in plotSubTree(). 
    
    Parameters
    ----------
    df_meta : DataFrame
        Dataframe containing meta information of individuals clustered, 
        including latitude and longitude under columns 'lat_lb' 'lon_lb'
    belonging_map : dict
        Map with node as key and list of individual indices clustered to the node as value
    color_int_map : dict
        Map with node as key and its color range denoted by [lb,ub] (a subset of [0,1]) as value
    G : networkx graph
        Subtree of focus
    map_bdry : tuple
        Tuple of format (llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon),
        where llcrnrlat, urcrnrlat, llcrnrlon, and urcrnrlon are matplotlib basemap keywords specifying 
        latitude of lower left hand corner, upper right hand corner, longitude of lower left hand corner, 
        and upper right hand corner of the desired map domain (degrees)        
    cmap : ColorMap
        Colormap used for the gradient coloring scheme 
    lat_lb : str
        Column name for latitudes in df_meta
    lon_lb : str
        Column name for longitudes in df_meta
    pt_size : int, optional 
        Scatter point size (default is 40)
    annot_lb : str, optional
        Column name for annotation labels in df_meta if provided (default is None)
    [Other optional arguments for plt.figure() function]:
        E.g. figsize=(9,6), dpi=100
    
    
    Returns
    -------
    fig
        Figure handle to the map plot
    """
    
    # default arguments
    
    lat_lb = kwargs.pop('lat_lb')
    lon_lb = kwargs.pop('lon_lb')
    pt_size = kwargs.pop('pt_size', 40)
    annot_lb = kwargs.pop('annot_lb', None)
    

    latitudes = np.array(df_meta[lat_lb].tolist())
    longitudes = np.array(df_meta[lon_lb].tolist())
    
    fig = plt.figure(**kwargs)
    
    # plot background
    
    m = Basemap(projection='merc', resolution='c',
                llcrnrlat=map_bdry[0], urcrnrlat=map_bdry[1],
                llcrnrlon=map_bdry[2], urcrnrlon=map_bdry[3], )
    m.drawcoastlines()
    m.fillcontinents(color='lightgray',lake_color='white')
    lats = m.drawparallels(np.linspace(-90, 90, 7),labels=[True,True,False,False],color=(0,0,0,0.1))
    lons = m.drawmeridians(np.linspace(-180, 180, 7),labels=[False,False,False,True],color=(0,0,0,0.1))


    # plot individuals in the subtree
    interested_clusters = set()
    for node in G.nodes():
        all_comms_at_level = belonging_map[node]
        interested_clusters |= set(all_comms_at_level)
        col = cmap((color_int_map[node][0]+color_int_map[node][1])/2)   
        m.scatter(df_meta.iloc[all_comms_at_level][lon_lb].tolist(), df_meta.iloc[all_comms_at_level][lat_lb].tolist(), latlon=True, 
                  c = [col], s=[pt_size], alpha=1, zorder=2)
    
    # annotate individuals
    if annot_lb:
        for ind_ID in interested_clusters:
            x,y = m(df_meta.iloc[ind_ID][lon_lb], df_meta.iloc[ind_ID][lat_lb])
            plt.text(x, y, df_meta.iloc[ind_ID][annot_lb],fontsize=5)

    # plot individuals not in the subtree
    col = (0,0,0,0.3)  #default: grey    
    for ind_ID in range(len(df_meta)):
        if ind_ID not in interested_clusters:
            m.scatter(df_meta.iloc[ind_ID][lon_lb], df_meta.iloc[ind_ID][lat_lb], latlon=True, 
                      c = [col], s=[pt_size], alpha=0.2, zorder=2) 

    return fig
    

def draw_pie(dist, xpos, ypos, size, cmap, size_cut, ax=None):
    """
    Draw pie chart at specified (x,y) position. 
    
    Parameters
    ----------
    dist : array
        Sorted array of the counts of categorical values of interest
    xpos : float
        x coordinate
    ypos : float
        y coordinate
    size : float
        Size of pie chart
    size_cut : float, optional
        Upper limit of size for pie chart at each node (default is 250)
    cmap : ColorMap
        Colormap used for categorical pie plot
    lat_lb : str
        Column name for latitudes in df_meta
    ax : axes handle
        Axes handle for pie chart, new figure created if not provided (default is None)    
    
    Returns
    -------
    ax : axes handle
        Axes handle to the pie chart
    colors : list
        List of colors in the pie charts
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,8))

    cumsum = np.cumsum(dist)
    cumsum = cumsum/cumsum[-1]
    pie = [0] + cumsum.tolist()
    ci = 0
    colors = list()
    
    pie_l = len(pie)-1
    for idx, r1, r2 in zip(range(pie_l),pie[:-1], pie[1:]):
        if idx==(pie_l-1):
            col = (0,0,0,0.3)
        else:
            col = cmap(ci)
        colors.append(col)
        
        angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2)
        x = [0] + np.cos(angles).tolist()
        y = [0] + np.sin(angles).tolist()

        xy = np.column_stack([x, y])       

        if size_cut<size: # limit pie size if node is too large
            ax.scatter([xpos], [ypos], marker=xy, alpha=0.2, s=size_cut, c=[col])
        else:
            ax.scatter([xpos], [ypos], marker=xy, s=size, c=[col])        

        ci += 1
        
    return ax, colors


def plotTreeWithPie(G_whole,belonging_map,df_meta,col_lb,top_num,cmap_pie,**kwargs):
    """
    Plot the tree where each node is a pie chart
    displaying the distribution of the specified categorical attribute for individuals clustered in node. 
    
    Parameters
    ----------
    G_whole : networkx graph
        The complete hierarchical structure tree
    belonging_map : dict
        Map with node as key and list of individual indices clustered to the node as value
    df_meta : DataFrame
        Dataframe containing meta information of individuals clustered, 
        including the categorical values used for pie charts specified by 'col_lb'
    col_lb : str
        Column name for categorical values of interest in df_meta
    top_num : int
        Number of top categories to explicitly show, where the rest are combined as 'others'
    cmap_pie : ColorMap
        Colormap used for categorical pie plot
    tree_aspect : float, optional
        Aspect ratio for tree plot (default is 1.0)
    size_cut : float, optional
        Upper limit of size for pie chart at each node (default is 250)
    size_scale : float, optional
        Scale ratio of node size, i.e., the plotted node has size cluster size * size_scale (default is 10)
    figsize : tuple, optional
        Argument for plt.figure() function (default is (19,8))
    dpi : int, optional
        Argument for plt.figure() function (default is 100)
    [Other optional arguments for plt.legend()) function]:
        E.g. ncol=2, loc="upper left"
    
    
    Returns
    -------
    fig
        Figure handle to the tree plot
    labels_cnt : named Series
        Counts of categorical values
    labels_top : named Series
        Counts of top categorical values 
    colors : list
        List of colors in the pie charts
    """
    
    # default arguments
    
    tree_aspect = kwargs.pop('tree_aspect', 1)
    size_cut = kwargs.pop('size_cut', 250)
    size_scale = kwargs.pop('size_scale', 10)
    figsize = kwargs.pop('figsize', (19,8))
    dpi = kwargs.pop('dpi', 100)
    
    # obtain label distribution 
    
    labels = np.unique(df_meta[col_lb])
    labels_cnt = df_meta[col_lb].value_counts()
                       
    df_agg =  labels_cnt.sort_values(ascending=False)
    top_list = df_agg.index.to_list()[:top_num]
    labels_top = pd.Series([labels_cnt[s] for s in top_list]+[sum(labels_cnt[~labels_cnt.index.isin(top_list)])],index=top_list+['others'])

    top_cnt = dict()
    for k in list(belonging_map.keys()):
        ind_in_node_cnt = df_meta.iloc[belonging_map[k]][col_lb].value_counts()
        node_cnt = sum(ind_in_node_cnt)
        fracs = [ind_in_node_cnt[l] if l in ind_in_node_cnt else 0 for l in labels]
        fracs_indexed = pd.Series(fracs,index=labels)
        fracs_top = [fracs_indexed[s] for s in top_list]+[sum(fracs_indexed[~fracs_indexed.index.isin(top_list)])]
        top_cnt[k] = fracs_top
    
    # plot tree
    
    fig = plt.figure(figsize=figsize,dpi=dpi)
    plt.box(False)
    plt.axis('off')
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
    pos_whole =graphviz_layout(G_whole, prog='dot')
    ax = plt.gca()
    ax.set_aspect(tree_aspect) 
    nx.draw_networkx_edges(G_whole,pos_whole,ax=ax,with_labels=False, arrows=True, node_size=50, alpha=0.9)

    # plot pie charts
    
    for chi in top_cnt.keys():

        x,y=pos_whole[chi]
        fracs = np.array(top_cnt[chi])
        sort_frac = np.argsort(-fracs)
        ax, colors = draw_pie(fracs, x, y, sum(fracs)*size_scale, 
                              cmap=cmap_pie, size_cut=size_cut, ax=ax)  

    patches = list()
    cate_info = ["%s (%d)" % (lab,labels_top[lab]) for lab in labels_top.index]
    for ci,c in enumerate(cate_info):
        patches.append(mpatches.Patch(color=colors[ci], label=c))

    plt.legend(handles=patches,**kwargs)
    
    return fig, labels_cnt, labels_top, colors


def plotLbDist(labels_cnt, labels_top, colors, xlab="", figsize=(12,3)):
    """
    Plot the overall distribution of the categorical attribute of interest in bar chart
    
    Parameters
    ----------
    labels_cnt : named Series
        Counts of categorical values
    labels_top : named Series
        Counts of top categorical values 
    colors : list
        List of colors used in the pie charts
    xlab : str, optional 
        x label of the plot (default is "")
    figsize : tuple, optional
        Argument for plt.figure() function (default is (12,3))
    
    Returns
    -------
    fig : figure handle
        Figure handle to the bar chart
    """
    
    fig = plt.figure(figsize=figsize) 
    plt.box(False)
    barlist=plt.bar(labels_cnt.index, labels_cnt.values, color=(0,0,0,0.3))
    for col,lb in zip(colors[:-1], labels_top.index.tolist()):
        barlist[np.where(labels_cnt.index==lb)[0][0]].set_color(col)
    plt.ylabel('number of individuals', size=13)
    plt.xlabel(xlab, size=13)
    plt.xticks(rotation=45)
    plt.xlim(-0.5,len(labels_cnt))
    
    return fig


def plotTreeWithMeanVal(G_whole,belonging_map,df_meta,col_lb,cmap,**kwargs):
    """
    Plot the tree where each node is colored by the mean value 
    of the specified attribute of interest for individuals clustered in node. 
    
    Parameters
    ----------
    G_whole : networkx graph
        The complete hierarchical structure tree
    belonging_map : dict
        Map with node as key and list of individual indices clustered to the node as value
    df_meta : DataFrame
        Dataframe containing meta information of individuals clustered, 
        including the attribute of interest specified by 'col_lb' 
    col_lb : str
        Column name for the attribute of interest in df_meta
    cmap : ColorMap
        Colormap used for plotting mean values
    tree_aspect : float, optional
        Aspect ratio for tree plot (default is 1.0)
    size_cut : float, optional
        Upper limit of size for pie chart at each node (default is 250)
    size_scale : float, optional
        Scale ratio of node size, i.e., the plotted node has size cluster size * size_scale (default is 10)
    color_min : float, optional
        Lower limit of colorbar value range (default is the minimum value of the attribute)
    color_scale : float, optional
        Length of colorbar value range (default is the maximum minus minimum value of the attribute)
    [Other optional arguments for plt.figure()) function]:
        E.g. figsize=(19,8), dpi=100
    
    
    Returns
    -------
    fig
        Figure handle to the tree plot
    fig_cb
        Figure handle to the corresponding colorbar
    """
    
    # default parameters
    
    tree_aspect = kwargs.pop('tree_aspect', 1.0)
    size_cut = kwargs.pop('size_cut', 250)
    size_scale = kwargs.pop('size_scale', 10)

    mean_map = dict()
    for c in list(belonging_map.keys()):
        ind_in_node_cnt = df_meta.loc[belonging_map[c]][col_lb].value_counts()
        mean_map[c] = np.mean(df_meta.loc[belonging_map[c]][col_lb])

    color_min = kwargs.pop('color_min', round((min(mean_map.values()))*2)/2)
    color_scale = kwargs.pop('color_scale', round((max(mean_map.values()) -  color_min )*2)/2)
    
    # plot tree
    
    fig = plt.figure(**kwargs)
    plt.box(False)
    plt.axis('off')
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
    pos_whole =graphviz_layout(G_whole, prog='dot')
    ax = plt.gca()
    ax.set_aspect(tree_aspect)
    nx.draw_networkx_edges(G_whole,pos_whole,ax=ax,with_labels=False, arrows=True, node_size=50, alpha=0.8)

    # color nodes
    for chi in mean_map.keys():

        x,y=pos_whole[chi]
        mean_val = mean_map[chi]

        if len(belonging_map[chi])*size_scale>size_cut:
            ax.scatter([x], [y], alpha=0.2, s=size_cut, c=[cmap((mean_val-color_min)/color_scale)], edgecolors=(0,0,0,0.3))
        else:
            ax.scatter([x], [y], alpha=1, s=len(belonging_map[chi])*size_scale, c=[cmap((mean_val-color_min)/color_scale)])
    
    # plot colorbar
    fig_cb = plt.figure(figsize=(3, 1.5))
    img = plt.imshow(np.array([[color_min,color_min+color_scale]]), cmap=cmap)
    img.set_visible(False)
    plt.gca().set_visible(False)
    cb = plt.colorbar(orientation="horizontal") 
    cb.ax.set_title('value')
    
    return fig, fig_cb