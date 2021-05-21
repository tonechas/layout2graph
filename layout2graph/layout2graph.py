"""AUTOMATIC ASSESSMENT OF FACILITY LAYOUTS

1. Create a target graph that models the specifications
   of a plant layout.
2. Launch AutoCAD (if necessary).
3. Open a layout drawing in .dwg format.
3. Automatically convert the layout into a graph.
4. Simplify the resulting graph.
5. Check that the layout is compliant with the
   specifications by comparing the simplified graph
   with the target graph.

Author: antfdez@uvigo.es
"""

import comtypes
import functools
import glob
import itertools
import logging
import math
import networkx as nx
import pathlib
import pickle
import os
import psutil
import sys
import win32com.client


# Necessary to import the remaining modules when running tests
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


from config import study_case, target_nodes, target_edges, key
from point_in_polygon import wn_PnPoly
import utils


#===========#
# Constants #
#===========#

# Layers
REQUIRED_LAYERS = {
    'NAMES', 
    'BORDERS', 
    'INNER_DOORS', 
    'OUTER_DOORS', 
    'OUTER_WALLS', 
    'INNER_WALLS', 
    'TEMP',
    }

VISIBLE_LAYERS = REQUIRED_LAYERS.difference(['NAMES'])

# AutoCAD codes
CENTER = 10
START_POINT = 10
END_POINT = 11
RADIUS = 40
START_ANGLE = 50
END_ANGLE = 51

PRECISION = 16


# AutoLISP scripts
explode_cmd = '\n'.join(
    ['(setq ssb (ssget "_x" (list (cons 8 "TEMP") (cons 0 "LWPOLYLINE"))))',
      '(setvar "qaflags" 1)',
      '(command "EXPLODE" ssb "")',
      '(setvar "qaflags" 0)',
      ]) + '\n'

erase_cmd = '\n'.join(
    ['(setq sstemp (ssget "_x" (list (cons 8 "TEMP"))))',
      '(command "ERASE" sstemp "")',
      ]) + '\n'


# Configuration for drawing graphs
props = dict(boxstyle="round", facecolor="w", alpha=0.5)


#===========================================#
# Helpers for file and directory management #
#===========================================#


def find_drawings(dirpath):
    """Recursively detect .dwg files contained in a folder.

    Parameters
    ----------
    dirpath : string
        Full path name of the folder that contains the drawings.

    Returns
    -------
    dwgfiles : list
        Full path names of all the drawing files in `dirpath` (and its
        subfolders).    
    """
    dwgfiles = [os.path.join(head, filename)
                for head, dirs, files in os.walk(dirpath)
                for filename in files
                if filename.endswith('.dwg')]
    return dwgfiles


#====================#
# Helpers for graphs #
#====================#


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


#============#
# Decorators #
#============#


def retry_decorator(func):
    """Decorator for avoiding com_error."""
    @functools.wraps(func)
    def wrapper_retry(*args, **kwargs):
        args_repr = [repr(a) for a in args]                      # 1
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  # 2
        signature = ", ".join(args_repr + kwargs_repr)           # 3

        for _ in range(5):
            try:
                value = func(*args, **kwargs)
                return value
                break
            except win32com.client.pythoncom.pywintypes.com_error:
                print(f'Error in {func.__name__}({signature}), retrying...')
    return wrapper_retry


#=========#
# Classes #
#=========#


class LayoutCAD:
    """Class that models a layout drawn in AutoCAD."""

    def __init__(self, dwg_path, acad):
        self.acad = acad
        self.dwg_path = dwg_path
        root, self.drawing = os.path.split(dwg_path)
        self.folder, _ = os.path.split(root)
        self.project, _ = os.path.splitext(self.drawing)
        self.dwg = self.get_drawing()
        self.logger = self.get_logger()
        self.ms = self.dwg.ModelSpace
        self.layers = self.prepare_layers()
        self.spaces = self.get_spaces()
        self.inner_doors, self.outer_doors = self.get_doors()
        self.borders = self.get_entities('BORDERS')
        self.links = self.get_links()

    def cleanup(self):
        # Delete files with AutoCAD  instances properties
        files = glob.glob(f'{self.folder}/temp/*.tmp')
        print('Removing files...')
        for f in files:
            os.remove(f)  

        if self.drawing in self.acad.Documents:
            # Close drawing
            self.dwg.Close()
        

    def get_logger(self):
        filename = f'{self.folder}/logs/{self.project}.log'
        logging.basicConfig(level=logging.INFO,
                            format='%(message)s',
                            filename=filename,
                            filemode='w',
                            force=True)

        formatter = logging.Formatter('%(message)s')
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)

        logger = logging.getLogger()
        logger.propagate = False
        logger.addHandler(console)
        logger.info(self.drawing + '\n' + '-'*len(self.drawing))
        return logger


    @retry_decorator
    def robust_send_command(self, command):
        self.dwg.SendCommand(command)


    def get_drawing(self):
        """Return AutoCAD drawing."""
        for doc in self.acad.Documents:
            if doc.Name == self.drawing:
                dwg = doc
                break
        else:
            try:
                dwg = self.acad.Documents.Open(dwg_path)
            except comtypes.COMError:
                raise FileNotFoundError(f"AutoCAD can't open {self.dwg_path}")
        #dwg.Activate()
        return dwg


    def prepare_layers(self):
        """Check if the drawing has all the required layers 
        and set-up the drawing environment.
        """
        # Read layers from the drawing and define required and visible layers
        layers = [layer.Name for layer in self.dwg.Layers]
        
        # Add TEMP if necessary and make it the current layer
        if 'TEMP' not in layers:
            self.robust_send_command('-LAYER New TEMP\n ')
        self.robust_send_command('-LAYER _ON TEMP\n ')        
        self.robust_send_command('CLAYER TEMP\n ')
    
        # Check whether the drawing has all the required layers
        for req in REQUIRED_LAYERS:
            if req not in layers:
                raise ValueError(f'Layer {req} is missing')
            
        # Make necessary layers visible, hide the rest
        for lyr in layers:
            status = ('_ON' if lyr in VISIBLE_LAYERS else '_OFF')
            self.robust_send_command(f'-LAYER {status} {lyr}\n ')
    
        # Further adjustments
        self.robust_send_command('_UCSICON OFF\n ')
        
        return layers
    
    
    def get_space_id(self, obj):
        """Return identifier of a Text/MText object. The decimal part 
        of the identifier must be between 1 and 9."""
        handle = obj.Handle
        if obj.ObjectName == 'AcDbMText':
            Property = 'Text'
        elif obj.ObjectName == 'AcDbText':
            Property = 'TextString'
        else:
            raise ValueError("'obj' is not 'AcDbMText' or 'AcDbText'")
        filename = f'{self.folder}/temp/{handle}.tmp'
        commands = [
            f'(setq fh (open "{filename}" "w"))',
            f'(setq ename (handent "{handle}"))',
            f'(princ (getpropertyvalue ename "{Property}") fh)',
            '(close fh) ',
            ]
        self.robust_send_command('\n'.join(commands))
        with open(filename, 'r') as fh:
            return str2num(fh.read())


    @retry_decorator
    def get_entities(self, layer, ename=None):
        """Return the entities of a given type that are located 
        on a given layer of the model space.
    
        Parameters
        ----------
        layer : str
            Name of the layer.
        ename : str or collection of strings (optional, default `None`)
            Target entity names. If no entity name is passed in, then all 
            the objects are retrieved.
    
        Returns
        -------
        entities : list
            List of detected entities on `layer`.
    
        Raises
        -------
        ValueError if the entity name passed in is invalid.
        """
        ms = self.ms
        if ename is None:
            entities = [obj for obj in ms if obj.Layer==layer]
        elif isinstance(ename, (str, bytes, bytearray)):
            entities = [obj for obj in ms 
                        if obj.Layer==layer and obj.ObjectName==ename]
        elif hasattr(ename, '__iter__'):
            entities = [obj for obj in ms 
                        if obj.Layer==layer and obj.ObjectName in ename]
        else:
            raise ValueError('Invalid entity name')
        return entities


    def get_spaces(self):
        """"Return a list of Text/MText objects representing spaces 
        ordered by their IDs."""
        objs = self.get_entities('NAMES', ('AcDbMText', 'AcDbText'))
        return sorted(objs, key=lambda x: self.get_space_id(x))


    def has_layer(self, layer):
        """Check whether an AutoCAD drawing has a certain layer.
    
        Parameters
        ----------
        layer : str
            Name of the layer.
    
        Returns
        -------
        bool
            `True` if `selg.dwg` has a the input layer , `False` otherwise.
        """
        return layer in [lyr.Name for lyr in self.dwg.Layers]


    def get_group_layer(self, group):
        """Return the layer a group belongs to."""
        layer = group.Item(0).Layer
        for index in range(1, group.Count):
            if group.Item(index).Layer != layer:
                msg = f'Items of group {index} must be in the same layer'
                raise ValueError(msg)
        return layer


    def get_doors(self):
        """Return the inner and outer doors."""
        inner_doors, outer_doors = [], []
        for group in self.dwg.Groups:
            layer = self.get_group_layer(group)
            if  layer == 'INNER_DOORS':
                inner_doors.append(group)
            elif layer == 'OUTER_DOORS':
                outer_doors.append(group)
            else:
                raise ValueError(f'Detected group in layer {layer}')
        return inner_doors, outer_doors


    def get_arc_extremes(self, obj):
        """Return the start and end points of an arc."""
        handle = obj.Handle
        if obj.ObjectName != 'AcDbArc':
            raise ValueError("'obj' no es un arco")
        filename = fr'{self.folder}/temp/{handle}.tmp'
        commands = [
            f'(setq fh (open "{filename}" "w"))',
            f'(setq ename (handent "{handle}"))',
            
            #'(princ (getpropertyvalue ename "StartPoint") fh)',
            f'(princ (rtos (cadr (assoc {CENTER} (entget ename))) 1 {PRECISION}) fh)',
            '(princ " " fh)',
            f'(princ (rtos (caddr (assoc {CENTER} (entget ename))) 1 {PRECISION}) fh)',
            '(princ " " fh)',
            
            #'(princ (getpropertyvalue ename "EndPoint") fh)',
            f'(princ (rtos (cdr (assoc {RADIUS} (entget ename))) 1 {PRECISION}) fh)',
            '(princ " " fh)',
            f'(princ (rtos (cdr (assoc {START_ANGLE} (entget ename))) 1 {PRECISION}) fh)',
            '(princ " " fh)',
            f'(princ (rtos (cdr (assoc {END_ANGLE} (entget ename))) 1 {PRECISION}) fh)',
            '(close fh) ',
            ]
        self.robust_send_command('\n'.join(commands))
        with open(filename, 'r') as fh:
            numbers = fh.read().split(' ')
        xc, yc, r, a0, a1 = [float(s) for s in numbers]
        
        x0 = xc + r*math.cos(a0)
        y0 = yc + r*math.sin(a0)
        x1 = xc + r*math.cos(a1)
        y1 = yc + r*math.sin(a1)
        start_point = (x0, y0, 0)
        end_point = (x1, y1, 0)
        return start_point, end_point


    @retry_decorator
    def get_line_extremes(self, obj):
        """Return the start and end points of a line."""
        handle = obj.Handle
        if obj.ObjectName != 'AcDbLine':
            raise ValueError("'obj' no es una línea")
        filename = fr'{self.folder}/temp/{handle}.tmp'
        commands = [
            f'(setq fh (open "{filename}" "w"))',
            f'(setq ename (handent "{handle}"))',
            
            f'(princ (rtos (cadr (assoc {START_POINT} (entget ename))) 1 {PRECISION}) fh)',
            '(princ " " fh)',
            f'(princ (rtos (caddr (assoc {START_POINT} (entget ename))) 1 {PRECISION}) fh)',
            '(princ " " fh)',

            f'(princ (rtos (cadr (assoc {END_POINT} (entget ename))) 1 {PRECISION}) fh)',
            '(princ " " fh)',
            f'(princ (rtos (caddr (assoc {END_POINT} (entget ename))) 1 {PRECISION}) fh)',
            '(close fh) ',
            ]
        self.robust_send_command('\n'.join(commands))
        with open(filename, 'r') as fh:
            numbers = fh.read().split(' ')
        x_start, y_start, x_end, y_end = [float(s) for s in numbers]
        start_point = x_start, y_start
        end_point = x_end, y_end
        return start_point, end_point


    def simplify_doors(self, doors):
        for door in doors:
            original_layer = self.get_group_layer(door)
            for index in range(door.Count):
                obj = door.Item(index)
                if obj.ObjectName != 'AcDbArc':
                    first_non_arc = obj
                    break
            for index in range(door.Count):
                obj = door.Item(index)
                if obj.ObjectName == 'AcDbArc':
                    start_point, end_point = self.get_arc_extremes(obj)
                    x_start, y_start, z_start = start_point
                    x_end, y_end, z_end = end_point
                    the_arc = obj
                    
                    self.robust_send_command(f'CLAYER {original_layer}\n ')
                    self.robust_send_command(f'LINE {x_start},{y_start} {x_end},{y_end}\n ')
                    chord = self.ms.Item(self.ms.Count - 1)
                    
                    self.robust_send_command(arc2chord(first_non_arc, chord, the_arc))
                    self.robust_send_command('CLAYER TEMP\n ')


    def get_links(self, info=True):
        self.simplify_doors(self.inner_doors)
        self.simplify_doors(self.outer_doors)
        links = self.inner_doors + self.borders
        if info:
            self.display_info()
        return links


    def display_info(self):
        """Show verbose output during execution."""
        self.logger.info('\nLAYERS\n------')
        for lyr in sorted(self.layers): 
            self.logger.info(lyr)
        
        self.logger.info('\nNAMES\n-----')
        for obj in self.spaces:
            number = self.get_space_id(obj)
            truncated = number if isinstance(number, int) else int(number)
            self.logger.info(f'{number}. {key[truncated]}')
        
        self.logger.info(f'\n#INNER_DOORS: {len(self.inner_doors)}')
        self.logger.info(f'#OUTER_DOORS: {len(self.outer_doors)}')
        self.logger.info(f'#BORDERS: {len(self.borders)}')


    def connected(self, link):
        """Determine the graph's edge corresponding to a door in the layout.
    
        Parameters
        ----------
        doc32: ****
            *******************.
        spaces : list
            List of AutoCAD text objects with the names of the layout spaces.
        link : AutoCAD object or `AcDbGroup`
            Either an AutoCAD object (typically an instance of `AcDbLine`, 
            `AcDbArc`, etc) or a group of objects, i.e. an instance of 
            `AcDbGroup`.
    
        Returns
        -------
        edge : tuple
            Tuple of two strings, representing a pair of spaces that 
            are connected through `door` in the layout
        """
    
        def cleanup(line, link):
            #_ = acadpy.best_interface(line).Delete()
            line.Delete()
            self.robust_send_command(erase_cmd)
            #doc.SendCommand(f'CHPROP (handent "{link.handle}") \n_LA BORDERS\n ')
            link.Visible = True
    
        link.Visible = False
        edge = None
    
        for index, start in enumerate(self.spaces[:-1]):
            #x_start, y_start, _ = start.InsertionPoint
            x_start, y_start = get_bb_center(start)
            self.robust_send_command(f'-BOUNDARY {x_start},{y_start}\n ')
            self.robust_send_command(explode_cmd)
            boundaries = self.get_entities('TEMP')
            #boundaries = get_entities_py(acadpy, 'TEMP')
    
            V = [] # Only works for lines
            for item in boundaries:
                starting_point, _ = self.get_line_extremes(item)
                x, y = starting_point
                V.append([x, y])
                
            for end in self.spaces[index+1:]:
                x_end, y_end = get_bb_center(end)
                P = (x_end, y_end)
                
                self.robust_send_command(f'LINE {x_start},{y_start} {x_end},{y_end}\n ')
                line = self.ms.Item(self.ms.Count - 1)
                line.Update()
                if wn_PnPoly(P, V) != 0:
                    edge = (self.get_space_id(start), self.get_space_id(end))
                    cleanup(line, link)
                    return edge
                #_ = acadpy.best_interface(line).Delete()
                line.Delete()
            self.robust_send_command(erase_cmd) # should avoid this
        #doc.SendCommand(f'CHPROP (handent "{link.handle}") \n_LA {orig_layer}\n ')
        link.Visible = True
    
        return edge
    
    
    def generate_graph(self):
        graph_complete = nx.Graph(name=f'{self.project}')
        graph_complete.add_nodes_from(
            [self.get_space_id(obj) for obj in self.spaces])
        
        self.logger.info('\nDetecting pairwise connections\n' + '-'*30)
        count_doors, count_borders = 0, 0
        self.robust_send_command('-LAYER _OFF OUTER_DOORS\n ')        
        
        for link in self.links:
            if link.ObjectName == 'AcDbGroup':
                head = f'Door {count_doors}'
                count_doors += 1
            else:
                head = f'Border {count_borders}'
                count_borders += 1
            edge = self.connected(link)
            if edge is not None:
                node1, node2 = edge
                self.logger.info(f'{head}:  {node1}  -->  {node2}')
                graph_complete.add_edge(node1, node2)
            else:
                self.logger.info()
        self.robust_send_command('(command "_.CLOSE" "_Y") ')
        return graph_complete


#===========#
# Functions #
#===========#


def str2num(s):
    """Convert string to int or float number.
    
    Parameters
    ----------
    s : string
        String representing a number.
    
    Returns
    -------
    Number (int or float)
    
    Raises
    ------
    TypeError
        If `s` is not a string.
    ValueError
        If the string does not represent a (float or int) number.
    """
    try:
        x = float(s)
        if x.is_integer():
            return int(x)
        else:
            return x
    except ValueError:
        raise ValueError("'s' does not represent a number (int or float)")


def extract_prj_name(dwg_path):
    """Extract the project name from the full drawing file name."""
    head, tail = os.path.split(dwg_path)
    project, _ = os.path.splitext(tail)
    return project


def arc2chord(first_non_arc, chord, the_arc):
    command = '\n'.join(
        [f'(setq firstnonarc (handent "{first_non_arc.Handle}"))', 
         f'(setq chord (handent "{chord.Handle}"))', 
         '(command "GROUPEDIT" firstnonarc "ADD" chord "")', 
         f'(setq thearc (handent "{the_arc.Handle}"))', 
         '(command "GROUPEDIT" firstnonarc "REMOVE" thearc "")', 
         '(command "ERASE" thearc "" )',
         ]) + '\n'
    return command


def get_bb_center(obj):
    """Return center coordinates of object bounding box."""
    ((x0, y0, z0), (x1, y1, z1)) = obj.GetBoundingBox()
    xc = (x0 + x1)/2
    yc = (y0 + y1)/2
    return xc, yc


def get_autocad_instance():
    """Return a running instance of AutoCAD Application.

    Returns
    -------
    acad : win32com.gen_py.???.IAcadApplication
        AutoCAD's COM object.
    """
    for p in psutil.process_iter():
        if p.name() == 'acad.exe':
            if p.is_running():
                print('AutoCAD is running already')
                break
    else:
        print('Launching AutoCAD...')

    prog_id='AutoCAD.Application'

    try:
        # Late-bound IDispatch
        #acad = win32com.client.dynamic.Dispatch(prog_id)
        #acad = win32com.client.Dispatch(prog_id)
        # Early-bound IDispatchg
        acad = win32com.client.gencache.EnsureDispatch(prog_id)
    #except win32com.client.pythoncom.pywintypes.com_error:
    except:
        raise
        
    acad.Visible = True
    return acad


def get_complete_graph(dwg_path):
    project = extract_prj_name(dwg_path)
    head, _ = os.path.splitext(dwg_path)
    saved_graph = os.path.join(head, '.pkl')
    try:
        with open(saved_graph, 'rb') as pklfile:
            print(f"Reading {project}'s graph...")
            graph_complete = pickle.load(pklfile)

    except FileNotFoundError:
        print(f'Generating graph for {project}...')
        obj = LayoutCAD(project, acad)
        graph_complete = obj.generate_graph()
        with open(saved_graph, 'wb') as pklfile:
            pickle.dump(graph_complete, pklfile, 
                        protocol=pickle.HIGHEST_PROTOCOL)
        obj.cleanup()

    return graph_complete


def prune(complete, essential, required):
    pruned = complete.copy()
    
    # Detect corridors
    corridors = set()
    for node in pruned.nodes:
        if int(node) == 20:
            corridors.add(node)

    # Merge distributed spaces
    merge = {node: int(node) 
              for node in pruned.nodes 
              if isinstance(node, float) and node not in corridors}
    _ = nx.relabel_nodes(pruned, merge, copy=False)      
    
    # Remove self loops
    for node in nx.nodes_with_selfloops(pruned):
        pruned.remove_edge(node, node)
    
    # Remove corridors
    for corridor in corridors:
        set_of_edges = set(pruned.edges)  # necessary to avoid side effects
        adj2corr = set()
        for (first, second) in set_of_edges:
            if corridor == first:
                adj2corr.add(second)
                if first in pruned.nodes:
                    pruned.remove_node(first)
            elif corridor == second:
                adj2corr.add(first)
                if second in pruned.nodes:
                    pruned.remove_node(second)
        for orig, dest in itertools.combinations(adj2corr, 2):
            pruned.add_edge(orig, dest)
                
    # Remove edges incident on non essential nodes
    edgeless = set(complete.nodes).difference(essential).difference(corridors).union(required)
    for (start, end) in list(pruned.edges):  # Beware of side effects
        if start in edgeless or end in edgeless:
            pruned.remove_edge(start, end)
    
    # Sort nodes lexicographically for proper representation
    out = nx.Graph()
    out.name = pruned.name
    out.add_nodes_from(sorted(pruned.nodes))
    out.add_edges_from(sorted(pruned.edges))
    return out


def display_report(simplified, target):
    print(f'{simplified.name} report')
    same_nodes = set(simplified.nodes) == set(target.nodes)
    same_edges = set(simplified.edges) == set(target.edges)
    if (same_nodes and same_edges):
        print('Layout is compliant with specifications\n')
    else:
        missing_nodes = [node for node in target.nodes 
                         if node not in simplified.nodes]
        missing_edges = [edge for edge in target.edges 
                         if edge not in simplified.edges]
        if missing_nodes:
            print(f'Missing nodes: {", ".join(map(repr, missing_nodes))}')
        if missing_edges:
            print(f'Missing edges: {", ".join(map(repr, missing_edges))}\n')

#==============#
# Main program #
#==============#
#%%
if __name__ == '__main__':
    
    # Define data folders
    parent = os.path.dirname(os.getcwd()).replace('\\', '/')
    dwg_folder = f'{parent}/{study_case}/drawings'
    graph_folder = f'{parent}/{study_case}/graphs'
    temp_folder = f'{parent}/{study_case}/temp'

    # Create directories if necessary
    for f in [graph_folder, temp_folder]:
        pathlib.Path(f).mkdir(parents=True, exist_ok=True)
    
    # Create target graph
    target_graph = create_target(target_nodes, target_edges, name='Target')
    essential_nodes = get_essential_nodes(target_graph)
    required_nodes = get_required_nodes(target_graph)
    utils.draw_target(graph_folder, target_graph, 
                      essential_nodes, required_nodes, 
                      key, props)
    
    # Get AutoCAD instance
    try:
        acad = get_autocad_instance()
    except:
        print('ERROR: Unable to launch AutoCAD\n'
              'Please, start AutoCAD manually')
        sys.exit(1)
    
    # Find drawing files
    drawing_files = find_drawings(dwg_folder)
    if not drawing_files:
        print(f'ERROR: no drawings found in\n{dwg_folder}')
        sys.exit(1)
    
    # Loop through drawings
    graphs = []
    for dwg_path in drawing_files:
        project = extract_prj_name(dwg_path)
        saved_graph = os.path.join(graph_folder, f'{project}.pkl')
        try:
            # Read graph from disk
            with open(saved_graph, 'rb') as pklfile:
                print(f"Reading {project}'s graph...")
                graph_complete = pickle.load(pklfile)
    
        except FileNotFoundError:
            print(f'Generating graph for {project}...')
            try:
                # Convert layout into graph
                obj = LayoutCAD(dwg_path, acad)
                graph_complete = obj.generate_graph()
                with open(saved_graph, 'wb') as pklfile:
                    pickle.dump(graph_complete, pklfile, 
                                protocol=pickle.HIGHEST_PROTOCOL)
                obj.cleanup()
            except Exception as ex:
                print('Exception catched')
                print(ex)
                sys.exit(1)
                
        # Simplify graph
        graph_simplified = prune(graph_complete, essential_nodes,
                                 required_nodes)
        
        # Draw solution
        utils.draw_solution(graph_folder, graph_complete, graph_simplified, 
                            essential_nodes, required_nodes, key, props)
        
        # Print report
        display_report(graph_simplified, target_graph)
        
        graphs.append((graph_complete, graph_simplified))