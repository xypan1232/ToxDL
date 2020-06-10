__author__ = 'jasper.zuallaert'
# Helper functions for reading the go.obo file, which is expected to be located in 'inputs/go.obo'
GO_OBO_FILE_LOCATION = 'inputs/go.obo'

# Returns a GO graph built by means of a dictionary, with each child term pointing to its parent terms
# The go.obo files is used, and an extra file can also be specified: indices_file. This file (obtained from the DeepGO
# paper) maps each GO term to the index.
# If specified, the dictionary will look something like:
# {0: [1, 2], 1:[3,4,5], 4:[6]}
# If not specified, the dictionary will keep the GO terms as strings
# {'GO:0003955':['GO:0024991','GO:0494811'], ...}
# Each key is the child term, and the value list contains all of its parents
def build_graph(indices_file = None):
    dependencies = {}
    id = None
    for line in open(GO_OBO_FILE_LOCATION):
        if line.startswith('id:'):
            if id != None:
                dependencies[id] = is_a
            id = line[4:].strip()
            is_a = []
        elif line.startswith('is_a:'):
            is_a.append(line.split('!')[0][6:].strip())

    if indices_file == None:
        return dependencies
    else:
        indices_dict = {}
        for line in open(indices_file):
            ind,term = line.split('-')
            ind,term = int(ind.strip()), term.strip()
            indices_dict[term] = ind

        new_dependencies = {}
        for k in indices_dict:
            if k in dependencies:
                new_dependencies[indices_dict[k]] = [indices_dict[x] for x in dependencies[k] if x in indices_dict]
            else:
                new_dependencies[indices_dict[k]] = []
        dependencies = new_dependencies
        return dependencies

# Returns a set that contains all children for a given term (including the grandchildren and further generations)
# The term is specified as a string 'GO:0009348', and so are the entries in the returned set.
def get_all_children_for_term(term):
    graph_dict = build_graph()
    full_set = set()
    term_list = [term]
    while term_list:
        parent_term = term_list.pop(0)
        if not parent_term in full_set:
            full_set.add(parent_term)
            children = [k for k in graph_dict if parent_term in graph_dict[k]]
            for child in children:
                if child not in full_set:
                    term_list.append(child)

    return full_set