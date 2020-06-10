__author__ = 'jasper.zuallaert'
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

# All plot_ functions are helper functions that draw a single letter
def plot_a(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
            [0.0, 0.0],
            [0.5, 1.0],
            [0.5, 0.8],
            [0.2, 0.0],
        ]),
        np.array([
            [1.0, 0.0],
            [0.5, 1.0],
            [0.5, 0.8],
            [0.8, 0.0],
        ]),
        np.array([
            [0.225, 0.45],
            [0.775, 0.45],
            [0.85, 0.3],
            [0.15, 0.3],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1, height])[None, :] * polygon_coords
                                                 + np.array([left_edge, base])[None, :]),
                                                facecolor=color, edgecolor=color))
def plot_nothing(ax, base, left_edge, height, color):
    pass
def plot_c(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(
        matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=0.7 * 1.3, height=0.7 * height,
                                   facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 1, base], width=1.0, height=height,
                                              facecolor='white', edgecolor='white', fill=True))
def plot_d(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [0.2, 1.0],
            [0.2, 0.0],
        ]),
        np.array([
            [0.2, 1.0],
            [0.4, 1.0],
            [0.6, 0.94],
            [0.7, 0.90],
            [0.9, 0.75],
            [0.97, 0.62],
            [1.0, 0.50],
            [0.8, 0.50],
            [0.77, 0.62],
            [0.7, 0.70],
            [0.6, 0.74],
            [0.4, 0.8],
            [0.2, 0.8],
        ]),
        np.array([
            [0.2, 1-1.0],
            [0.4, 1-1.0],
            [0.6, 1-0.94],
            [0.7, 1-0.90],
            [0.9, 1-0.75],
            [0.97,1- 0.62],
            [1.0, 1-0.50],
            [0.8, 1-0.50],
            [0.77,1- 0.62],
            [0.7, 1-0.70],
            [0.6, 1-0.74],
            [0.4, 1-0.8],
            [0.2, 1-0.8],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1, height])[None, :] * polygon_coords
                                                 + np.array([left_edge, base])[None, :]),
                                                facecolor=color, edgecolor=color))
def plot_e(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.2],
            [0.2, 0.2],
            [0.2, 0.4],
            [0.8, 0.4],
            [0.8, 0.6],
            [0.2, 0.6],
            [0.2, 0.8],
            [1.0, 0.8],
            [1.0, 1.0],
            [0.0, 1.0],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1, height])[None, :] * polygon_coords
                                                 + np.array([left_edge, base])[None, :]),
                                                facecolor=color, edgecolor=color))
def plot_f(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
            [0.0, 0.0],
            [0.2, 0.0],
            [0.2, 0.2],
            [0.2, 0.4],
            [0.8, 0.4],
            [0.8, 0.6],
            [0.2, 0.6],
            [0.2, 0.8],
            [1.0, 0.8],
            [1.0, 1.0],
            [0.0, 1.0],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1, height])[None, :] * polygon_coords
                                                 + np.array([left_edge, base])[None, :]),
                                                facecolor=color, edgecolor=color))
def plot_g(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(
        matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=0.7 * 1.3, height=0.7 * height,
                                   facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 1, base], width=1.0, height=height,
                                              facecolor='white', edgecolor='white', fill=True))
    ax.add_patch(
        matplotlib.patches.Rectangle(xy=[left_edge + 0.825, base + 0.085 * height], width=0.174, height=0.415 * height,
                                     facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(
        matplotlib.patches.Rectangle(xy=[left_edge + 0.625, base + 0.35 * height], width=0.374, height=0.15 * height,
                                     facecolor=color, edgecolor=color, fill=True))
def plot_h(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
            [0.0, 0.0],
            [0.2, 0.0],
            [0.2, 0.4],
            [0.8, 0.4],
            [0.8, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.8, 1.0],
            [0.8, 0.6],
            [0.2, 0.6],
            [0.2, 1.0],
            [0.0, 1.0],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1, height])[None, :] * polygon_coords
                                                 + np.array([left_edge, base])[None, :]),
                                                facecolor=color, edgecolor=color))
def plot_i(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
            [0.4, 0.0],
            [0.4, 0.0],
            [0.6, 0.0],
            [0.6, 1.0],
            [0.4, 1.0],
        ]),
        np.array([
            [0.2, 0.0],
            [0.8, 0.0],
            [0.8, 0.1],
            [0.2, 0.1],
        ]),
        np.array([
            [0.2, 1-0.0],
            [0.8, 1-0.0],
            [0.8, 1-0.1],
            [0.2, 1-0.1],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1, height])[None, :] * polygon_coords
                                                 + np.array([left_edge, base])[None, :]),
                                                facecolor=color, edgecolor=color))
def plot_k(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
            [0.0, 0.0],
            [0.2, 0.0],
            [0.2, 0.4],
            [0.9, 0.0],
            [1.0, 0.1],
            [0.3, 0.5],
            [1.0, 0.9],
            [0.9, 1.0],
            [0.2, 0.6],
            [0.2, 1.0],
            [0.0, 1.0],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1, height])[None, :] * polygon_coords
                                                 + np.array([left_edge, base])[None, :]),
                                                facecolor=color, edgecolor=color))
def plot_l(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.2],
            [0.2, 0.2],
            [0.2, 1.0],
            [0.0, 1.0],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1, height])[None, :] * polygon_coords
                                                 + np.array([left_edge, base])[None, :]),
                                                facecolor=color, edgecolor=color))
def plot_m(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
            [0.0, 0.0],
            [0.2, 0.0],
            [0.2, 0.7],
            [0.5, 0.3],
            [0.8, 0.7],
            [0.8, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.8, 1.0],
            [0.5, 0.7],
            [0.2, 1.0],
            [0.0, 1.0],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1, height])[None, :] * polygon_coords
                                                 + np.array([left_edge, base])[None, :]),
                                                facecolor=color, edgecolor=color))
def plot_n(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
            [0.0, 0.0],
            [0.2, 0.0],
            [0.2, 0.8],
            [0.8, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.8, 1.0],
            [0.8, 0.2],
            [0.2, 1.0],
            [0.0, 1.0],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1, height])[None, :] * polygon_coords
                                                 + np.array([left_edge, base])[None, :]),
                                                facecolor=color, edgecolor=color))
def plot_p(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
            [0.0, 0.0],
            [0.2, 0.0],
            [0.2, 1.0],
            [0.0, 1.0],
        ]),
        np.array([
            [0.2, 1.0],
            [0.4, 1.0],
            [0.6, 0.97],
            [0.7, 0.96],
            [0.8, 0.93],
            [0.9, 0.87],
            [1.0, 0.77],
            [1.0, 0.75],
            [0.9, 0.75],
            [0.8, 0.82],
            [0.7, 0.86],
            [0.6, 0.87],
            [0.5, 0.88],
            [0.4, 0.9],
            [0.2, 0.9],
        ]),
        np.array([
            [0.2,1-1.0+0.5],
            [0.4,1-1.0+0.5],
            [0.6,1-0.97+0.5],
            [0.7,1-0.96+0.5],
            [0.8,1-0.93+0.5],
            [0.9,1-0.87+0.5],
            [1.0,1-0.77+0.5],
            [1.0,1-0.75+0.5],
            [0.9,1-0.75+0.5],
            [0.8,1-0.82+0.5],
            [0.7,1-0.86+0.5],
            [0.6,1-0.87+0.5],
            [0.5,1-0.88+0.5],
            [0.4,1-0.9+0.5],
            [0.2,1-0.9+0.5],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1, height])[None, :] * polygon_coords
                                                 + np.array([left_edge, base])[None, :]),
                                                facecolor=color, edgecolor=color))
def plot_r(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
            [0.0, 0.0],
            [0.2, 0.0],
            [0.2, 1.0],
            [0.0, 1.0],
        ]),
        np.array([
            [0.2, 0.4],
            [0.9, 0.0],
            [1.0, 0.1],
            [0.2, 0.6],
        ]),
        np.array([
            [0.2, 1.0],
            [0.4, 1.0],
            [0.6, 0.97],
            [0.7, 0.96],
            [0.8, 0.93],
            [0.9, 0.87],
            [1.0, 0.77],
            [1.0, 0.75],
            [0.9, 0.75],
            [0.8, 0.82],
            [0.7, 0.86],
            [0.6, 0.87],
            [0.5, 0.88],
            [0.4, 0.9],
            [0.2, 0.9],
        ]),
        np.array([
            [0.2,1-1.0+0.5],
            [0.4,1-1.0+0.5],
            [0.6,1-0.97+0.5],
            [0.7,1-0.96+0.5],
            [0.8,1-0.93+0.5],
            [0.9,1-0.87+0.5],
            [1.0,1-0.77+0.5],
            [1.0,1-0.75+0.5],
            [0.9,1-0.75+0.5],
            [0.8,1-0.82+0.5],
            [0.7,1-0.86+0.5],
            [0.6,1-0.87+0.5],
            [0.5,1-0.88+0.5],
            [0.4,1-0.9+0.5],
            [0.2,1-0.9+0.5],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1, height])[None, :] * polygon_coords
                                                 + np.array([left_edge, base])[None, :]),
                                                facecolor=color, edgecolor=color))
def plot_t(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 0.4, base],
                                              width=0.2, height=height, facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base + 0.8 * height],
                                              width=1.0, height=0.2 * height, facecolor=color, edgecolor=color,
                                              fill=True))
def plot_q(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
            [0.55, 0.23],
            [0.68, 0.32],
            [0.92, 0.13],
            [0.8, 0.03],
        ])
    ]
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge + 0.5, base + 0.5 * height], width=1.0, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(
        matplotlib.patches.Ellipse(xy=[left_edge + 0.5, base + 0.5 * height], width=0.7, height=0.7 * height,
                                   facecolor='white', edgecolor='white'))
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1, height])[None, :] * polygon_coords
                                                 + np.array([left_edge, base])[None, :]),
                                                facecolor=color, edgecolor=color))
def plot_v(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
            [0.0, 0.9],
            [0.1, 1.0],
            [0.5, 0.2],
            [0.9, 1.0],
            [1.0, 0.9],
            [0.6, 0.0],
            [0.4, 0.0],
        ]),
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1, height])[None, :] * polygon_coords
                                                 + np.array([left_edge, base])[None, :]),
                                                facecolor=color, edgecolor=color))
def plot_w(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
            [0.0, 0.9],
            [0.1, 1.0],
            [0.5, 0.2],
            [0.4, 0.0],
        ]),
        np.array([
            [1-0.0, 0.9],
            [1-0.1, 1.0],
            [1-0.5, 0.2],
            [1-0.4, 0.0],
        ]),
        np.array([
            [0.5, 0.4],
            [0.32, 0.2],
            [0.5, 0.1],
            [0.68, 0.2],
            # [0.5, 0.4],
        ]),
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1, height])[None, :] * polygon_coords
                                                 + np.array([left_edge, base])[None, :]),
                                                facecolor=color, edgecolor=color))
def plot_y(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
            [0.0, 0.9],
            [0.1, 1.0],
            [0.5, 0.75],
            [0.9, 1.0],
            [1.0, 0.9],
            [0.6, 0.6],
            [0.6, 0.0],
            [0.4, 0.0],
            [0.4, 0.6],
        ]),
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1, height])[None, :] * polygon_coords
                                                 + np.array([left_edge, base])[None, :]),
                                                facecolor=color, edgecolor=color))
def plot_s(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.6],
            [0.2, 0.6],
            [0.2, 0.8],
            [1.0, 0.8],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.4],
            [0.8, 0.4],
            [0.8, 0.2],
            [0.0, 0.2],
        ]),
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1, height])[None, :] * polygon_coords
                                                 + np.array([left_edge, base])[None, :]),
                                                facecolor=color, edgecolor=color))

# The colours for each amino acid
colors = ['lightsalmon', # A
          'lightseagreen', # C
          'red', # D
          'firebrick', # E
          'lime', # F
          'yellow', # G
          'maroon', # H
          'darkgoldenrod', # I
          'saddlebrown', # K
          'sandybrown', # L
          'thistle', # M
          'teal', # N
          'dodgerblue', # P
          'slateblue', # Q
          'darkred', # R
          'blue', # S
          'cyan', # T
          'gold', # V
          'limegreen', # W
          'darkgreen'] # Y
default_colors = {i: colors[i] for i in range(20)}
default_plot_funcs = {0: plot_a,
                      1: plot_c,
                      2: plot_d,
                      3: plot_e,
                      4: plot_f,
                      5: plot_g,
                      6: plot_h,
                      7: plot_i,
                      8: plot_k,
                      9: plot_l,
                      10: plot_m,
                      11: plot_n,
                      12: plot_p,
                      13: plot_q,
                      14: plot_r,
                      15: plot_s,
                      16: plot_t,
                      17: plot_v,
                      18: plot_w,
                      19: plot_y}

def _plot_weights_given_ax(ax, array,
                           height_padding_factor,
                           length_padding,
                           subticks_frequency,
                           highlight,
                           colors=default_colors,
                           plot_funcs=default_plot_funcs):
    if len(array.shape) == 3:
        array = np.squeeze(array)
    assert len(array.shape) == 2, array.shape
    if (array.shape[0] == 20 and array.shape[1] != 20):
        array = array.transpose(1, 0)
    assert array.shape[1] in (20,21)
    max_pos_height = 0.0
    min_neg_height = 0.0
    heights_at_positions = []
    depths_at_positions = []
    for i in range(array.shape[0]):
        # sort from smallest to highest magnitude
        acgt_vals = sorted(enumerate(array[i, :]), key=lambda x: abs(x[1]))
        positive_height_so_far = 0.0
        negative_height_so_far = 0.0
        for letter in acgt_vals:
            plot_func = plot_funcs[letter[0]]
            color = colors[letter[0]]
            if (letter[1] > 0):
                height_so_far = positive_height_so_far
                positive_height_so_far += letter[1]
            else:
                height_so_far = negative_height_so_far
                negative_height_so_far += letter[1]
            plot_func(ax=ax, base=height_so_far, left_edge=i, height=letter[1], color=color)
        max_pos_height = max(max_pos_height, positive_height_so_far)
        min_neg_height = min(min_neg_height, negative_height_so_far)
        heights_at_positions.append(positive_height_so_far)
        depths_at_positions.append(negative_height_so_far)

    # now highlight any desired positions; the key of
    # the highlight dict should be the color
    for color in highlight:
        for start_pos, end_pos in highlight[color]:
            assert start_pos >= 0.0 and end_pos <= array.shape[0]
            min_depth = np.min(depths_at_positions[start_pos:end_pos])
            max_height = np.max(heights_at_positions[start_pos:end_pos])
            ax.add_patch(
                matplotlib.patches.Rectangle(xy=[start_pos, min_depth],
                                             width=end_pos - start_pos,
                                             height=max_height - min_depth,
                                             edgecolor=color, fill=False))

    ax.set_xlim(-length_padding, array.shape[0] + length_padding)
    ax.xaxis.set_ticks(np.arange(0.0, array.shape[0] + 1, subticks_frequency))
    height_padding = max(abs(min_neg_height) * (height_padding_factor),
                         abs(max_pos_height) * (height_padding_factor))
    ax.set_ylim(min_neg_height - height_padding, max_pos_height + height_padding)

def _visualizeScores(array,
                     filename,
                     figsize=(100, 6),
                     height_padding_factor=0.2,
                     length_padding=1.0,
                     subticks_frequency=5.0,
                     colors=default_colors,
                     plot_funcs=default_plot_funcs,
                     highlight={}):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    _plot_weights_given_ax(ax=ax, array=array,
                           height_padding_factor=height_padding_factor,
                           length_padding=length_padding,
                           subticks_frequency=subticks_frequency,
                           colors=colors,
                           plot_funcs=plot_funcs,
                           highlight=highlight)
    plt.savefig(filename)
    plt.close('all')

# Function to call in this file. You need to specify an inputfile, containing the saliency maps, and an output directory,
# where the png image files will be stored
def visualizeSaliencyMapFile(inputfile, outputDir):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    lines = open('{}'.format(inputfile)).readlines()
    cnts = {'TN':0,'TP':0,'FN':0,'FP':0}

    pr = ''
    for i in range(len(lines)//3):
        firstLine_split_dash = lines[3 * i].strip().split(' - ')
        if len(firstLine_split_dash) == 2:
            seqname = firstLine_split_dash[1].strip()
        else:
            seqname = None

        l1 = [float(x) for x in lines[3*i].strip().split(',')[:2]]
        l2 = lines[3*i+1].strip().split(',')
        l3 = lines[3*i+2].strip().split(',')

        if   l1[0] >= 0.5 and l1[1] == 1:
            pr = 'TP'
        elif l1[0] >= 0.5 and l1[1] == 0:
            pr = 'FP'
        elif l1[0] <  0.5 and l1[1] == 1:
            pr = 'FN'
        elif l1[0] <  0.5 and l1[1] == 0:
            pr = 'TN'

        # hard-coding here: specify if you want to visualize sequences that are TP, FP, FN and/or TN (currently: all)
        if pr in ('TP','FP','FN','TN'):
        # hard-coding here: at most 200 images of the same sort (TP, TN, ...) will be created.
            if cnts[pr] < 200:
                if seqname:
                    filename = outputDir+'{}_{}_{:03d}_{:1.5f}_{:1d}.png'.format(seqname,pr,cnts[pr],l1[0],int(l1[1]))
                else:
                    filename = outputDir+'{}_{:03d}_{:1.5f}_{:1d}.png'.format(pr,cnts[pr],l1[0],int(l1[1]))
                fill = np.zeros((len(l2),20))

                for e, (c, v) in enumerate(zip(l2,l3)):
                    fill[e]['ACDEFGHIKLMNPQRSTVWY'.index(c)] = float(v)
                print('Visualizing {}'.format(filename))
                _visualizeScores(fill, filename)
                cnts[pr] += 1
