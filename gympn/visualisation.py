import simpn.visualisation
from simpn.visualisation import *

class Visualisation(simpn.visualisation.Visualisation):
    def __init__(self, sim_problem, layout_file=None):
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption('Petri Net Visualisation')
        assets.create_assets(assets.images, "assets")
        icon = pygame.image.load('./assets/logo.png')
        pygame.display.set_icon(icon)

        self.__running = False
        self._problem = sim_problem
        self._nodes = dict()
        self._edges = []
        self._selected_nodes = None

        # Add visualizations for prototypes, places, and transitions,
        # but not for places and transitions that are part of prototypes.
        element_to_prototype = dict()  # mapping of prototype element ids to prototype ids
        viznodes_with_edges = []
        for prototype in self._problem.prototypes:
            if prototype.visualize:
                prototype_viznode = prototype.get_visualisation()
                self._nodes[prototype.get_id()] = prototype_viznode
                viznodes_with_edges.append(prototype_viznode)
                for event in prototype.events:
                    element_to_prototype[event.get_id()] = prototype.get_id()
                if hasattr(prototype, 'actions'):
                    for action in prototype.actions:
                        element_to_prototype[action.get_id()] = prototype.get_id()
                for place in prototype.places:
                    element_to_prototype[place.get_id()] = prototype.get_id()
        for var in self._problem.places:
            if var.visualize and var.get_id() not in element_to_prototype:
                self._nodes[var.get_id()] = var.get_visualisation()
        for event in self._problem.events:
            if event.visualize and event.get_id() not in element_to_prototype:
                event_viznode = event.get_visualisation()
                self._nodes[event.get_id()] = event_viznode
                viznodes_with_edges.append(event_viznode)
        for action in self._problem.actions:
            if action.visualize and action.get_id() not in element_to_prototype:
                event_viznode = action.get_visualisation()
                self._nodes[action.get_id()] = event_viznode
                viznodes_with_edges.append(event_viznode)
                # Add visualization for edges.
        # If an edge is from or to a prototype element, it must be from or to the prototype itself.
        for viznode in viznodes_with_edges:
            for incoming in viznode._model_node.incoming:
                node_id = incoming.get_id()
                if node_id.endswith(".queue"):
                    node_id = node_id[:-len(".queue")]
                if node_id in element_to_prototype:
                    node_id = element_to_prototype[node_id]
                if node_id in self._nodes:
                    other_viznode = self._nodes[node_id]
                    self._edges.append(Edge(start=(other_viznode, Hook.RIGHT), end=(viznode, Hook.LEFT)))
            for outgoing in viznode._model_node.outgoing:
                node_id = outgoing.get_id()
                if node_id.endswith(".queue"):
                    node_id = node_id[:-len(".queue")]
                if node_id in element_to_prototype:
                    node_id = element_to_prototype[node_id]
                if node_id in self._nodes:
                    other_viznode = self._nodes[node_id]
                    self._edges.append(Edge(start=(viznode, Hook.RIGHT), end=(other_viznode, Hook.LEFT)))
        layout_loaded = False
        if layout_file is not None:
            try:
                self.__load_layout(layout_file)
                layout_loaded = True
            except FileNotFoundError as e:
                print("WARNING: could not load the layout because of the exception below.\nauto-layout will be used.\n",
                      e)
        if not layout_loaded:
            self.__layout()

        self.__screen = pygame.display.set_mode(self._size, pygame.RESIZABLE)
        self._buttons = self.__init_buttons()
        