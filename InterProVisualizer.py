__author__ = 'jasper.zuallaert'
# This python script combines+visualizes the outputs from interproscan and integratedgradients
# It will be called from SingleTermWorkflow.py
# input: - file as outputted by InterProScan              (which should've been run only on positive samples if we want to visualize only positive sequences)
#        - file as outputted by IntegratedGradientsRunner (+ PosSeqFromSaliencyMapFile if we want to visualize only positive sequences)
# output: to the desired output location, jpg files will be written, with some stuff:
# - On top, the protein sequence is located
# - Then, we show bars of the region that were found by interproscan
#   Each line contains one bar over a region, with also the information of the annotation at the beginning
#       if grey: this region was not annotated with the GO term we are looking for
#       if red: this region was annotated with the GO term
#       if orange: this region was annotated with one of the children of this GO term
#       (TODO: parents as well but not supported yet)
#   Finally, we also list the predicted probability of positive classification, as well as the saliency map (without scale for now)

# <interpro_file> each line (tab-separated) like:
# seq014  292b5f1ee59ff2bf77e146d061125793        75      SUPERFAMILY     SSF57038                50      75      3.79E-10        T       18-01-2019      IPR036146       Cyclotide superfamily   GO:0006952
# <fasta id> <some other id> <seq len> <domain/family/pattern type> <name> <start pos> <end pos> <dunno> <date> <dunno dunno dunno> <associated GO terms>

# <saliency_map_file> each three lines like:
# 0.39808332920074463,1,actual_length=19 - seq001
# G,F,K,D,L,L,K,G,A,A,K,A,L,V,K,T,V,L,F
# 0.02,0.05,0.07,0.03,0.02,0.06,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0



# Parameter that sets how wide the column with text at the left hand side should be
LEFT_COLUMN_WIDTH = 95
import turtle as t
import os
from GO_Graph_Builder import get_all_children_for_term

# Run the visualization script with parameters:
# - interpro_file: the location of the interproscan output file
# - saliency_map_file: the location of the saliency map file
# - term: the GO term we are interested in
# - outputDir: the output directory where all jpg images should be located
def runInterProVisualizer(interpro_file, saliency_map_file,term,outputDir):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    t.tracer(0)
    t.speed('fastest')

    GO_TERMS = [term]
    ALL_CHILD_TERMS = get_all_children_for_term(term)
    all_interpro = {}
    file_id = saliency_map_file.split('/')[-1].split('.')[0]
    saliency_lines = open(saliency_map_file).readlines()

    ### Read and store all interpro lines
    for line in open(interpro_file):
        line = line.strip().split('\t')
        fasta_id = line[0]
        record = (line[3]+' '+line[4], int(line[6]), int(line[7]), line[-1] if line[-1].startswith('GO') else '')

        if fasta_id in all_interpro:
            all_interpro[fasta_id].append(record)
        else:
            all_interpro[fasta_id] = [record]

    ### For all lines in the saliency map file, prepare the drawing
    to_draw = []
    for recN in range(len(saliency_lines)//3):
        line1 = saliency_lines[recN * 3 + 0].strip()
        line2 = saliency_lines[recN * 3 + 1].strip()
        line3 = saliency_lines[recN * 3 + 2].strip()

        fasta_id = line1.split('-')[-1].strip()
        prediction = float(line1.split(',')[0])
        sequence = line2.replace(',','')
        saliency_map = [float(x) for x in line3.split(',')]
        interpros = all_interpro[fasta_id] if fasta_id in all_interpro else []

        to_draw.append((fasta_id, prediction, sequence, saliency_map, interpros))

    ### Draw!
    for s in to_draw:
        fasta_id, prediction, sequence, saliency_map, interpros = s
        print('Drawing {fasta_id}')
        draw(file_id,
             fasta_id,
             sequence,
             interpros,
             saliency_map,
             prediction,
             50+35*(5+len(interpros)+1),
             11*len(sequence)+10+LEFT_COLUMN_WIDTH*11,
             GO_TERMS,
             ALL_CHILD_TERMS,
             outputDir)

########################################################################################
############################### DRAWING FUNCTIONS ######################################
########################################################################################
# Helper function, draws one record from the interproscan/saliencymap file
# - file_id: Name of the saliency file, for jpg file naming purposes
# - fasta_name: The uniprot id as specified
# - sequence: The protein sequence
# - list_of_interpro: List of tuples of format: (domain/family/... type, start_pos, stop_pos, GO term string)
# - saliency_map: a list of floats
# - prediction: a float with the predicted probability
# - h: the height of the drawing window
# - w: the height of the drawing window
# - GO_TERMS: The term we are looking for (coloured red)
# - ALL_CHILD_TERMS: The children of the term we are looking for (coloured orange)
# - outputDir: the output directory where all files should be stored
def draw(file_id,fasta_name, sequence, list_of_interpro, saliency_map, prediction, h, w, GO_TERMS, ALL_CHILD_TERMS, outputDir):
    t.Turtle()
    t.hideturtle()

    t.screensize(w,h)
    t.setup(width=w, height=h, startx=None, starty=None)
    t.setworldcoordinates(0,-h,w,0)

    drawGrid(h,w)

    put_text(fasta_name, 0, 1)
    put_text(sequence, 0, LEFT_COLUMN_WIDTH)

    # colormap = {'SUP':'red','pFa':'blue','Pro':'dark green'}
    for ln,interpro_tup in enumerate(list_of_interpro):
        color = 'red' if any(x in interpro_tup[-1] for x in GO_TERMS) else \
            'orange' if any(x in interpro_tup[-1] for x in ALL_CHILD_TERMS) else \
            'grey'
        put_text(interpro_tup[3] + ' ' + interpro_tup[0], ln+1, 1 if interpro_tup[3] else 0)
        put_rectangle(ln+1,LEFT_COLUMN_WIDTH+interpro_tup[1],LEFT_COLUMN_WIDTH+interpro_tup[2],color=color)

    put_text('saliency map',len(list_of_interpro)+1,1)
    put_text('prediction: {prediction}',len(list_of_interpro)+2,1)

    if not all(x == 0.0 for x in saliency_map):
        put_graph(saliency_map,len(list_of_interpro)+1)
    cnv = t.getcanvas()
    cnv.postscript(file="{outputDir}/tmp.eps")
    t.resetscreen()
    os.system('convert {outputDir}/tmp.eps {outputDir}/{file_id}_{fasta_name}.jpg ; rm {outputDir}/tmp.eps')

# Draws the supporting background grid
def drawGrid(h,w):
    t.color('#EFEFEF')
    for x in range(LEFT_COLUMN_WIDTH*11,w,11):
        t.penup()
        t.goto(x,0)
        t.pendown()
        t.goto(x,-h)
    for y in range(25,h,35):
        t.penup()
        t.goto(LEFT_COLUMN_WIDTH,-y)
        t.pendown()
        t.goto(w,-y)

# draws text at the given coordinates
# line = line number (pixels get counted automatically)
# x = character position along the x-axis (pixels counted automatically)
# if x == 1 (left column), text gets clipped to remain within the column width
def put_text(text, line, x):
    t.color('black')
    if ' ' in text:
        text = text[:LEFT_COLUMN_WIDTH-1]

    for i,letter in enumerate(text):
        t.penup()
        t.goto((x+i)*11, -line*35-25)
        t.write(letter, font=('Courier New',20,'normal'))

# draws a rectangle from left to right, for two given positions in the sequence
# x_start and x_stop = position in the sequence (x_stop included in the rectangle)
def put_rectangle(line, x_start, x_stop, color):
    t.color(color)
    t.penup()
    t.goto(x_start*11, -line*35-25+5)
    t.pendown()
    t.begin_fill()
    t.goto(x_stop*11, -line*35-25+5)
    t.goto(x_stop*11, -line*35-25+25)
    t.goto(x_start*11, -line*35-25+25)
    t.goto(x_start*11, -line*35-25+5)
    t.end_fill()

# draws a graph with the saliency_values
# the graph takes 5 lines
# n_of_lines_before = the number of lines that came before (= 1 + the amount of interproscan findings)
def put_graph(saliency_values, n_of_lines_before):
    t.color('light blue')
    sal_max, sal_min = max(saliency_values+[0]), min(saliency_values+[0])
    div_factor = (abs(sal_min) + sal_max) / (35*5)
    saliency_values = [x / div_factor for x in saliency_values]
    zero_point_y = -n_of_lines_before*35-25-max(saliency_values+[0])

    for i,value in enumerate(saliency_values):
        t.penup()
        t.goto((i+LEFT_COLUMN_WIDTH)*11+1  ,zero_point_y)
        t.pendown()
        t.begin_fill()
        t.goto((i+LEFT_COLUMN_WIDTH+1)*11-1,zero_point_y)
        t.goto((i+LEFT_COLUMN_WIDTH+1)*11-1,zero_point_y+value)
        t.goto((i+LEFT_COLUMN_WIDTH)*11  +1,zero_point_y+value)
        t.goto((i+LEFT_COLUMN_WIDTH)*11  +1,zero_point_y)
        t.end_fill()


