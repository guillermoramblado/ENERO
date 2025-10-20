import gym
import numpy as np
import networkx as nx
import random
from gym import error, spaces, utils
from random import choice
import pandas as pd
import pickle
import json 
import os.path
import gc
import defo_process_results as defoResults
import matplotlib.pyplot as plt
import tensorflow as tf

class Env16(gym.Env):
    """
    Here I only take X% of the demands. There are some flags
    that indicate if to take the X% larger demands, the X% from the 5 most loaded links
    or random.

    Environment used in the middlepoint routing problem. Here we compute the SP to reach a middlepoint.
    We are using bidirectional links in this environment!
    In this environment we make the MP between edges.

    #Realmente, [0] almacena el ancho de banda que está alojando actualmente el enlace, no su uso.
    #Para obtener realmente el uso, será necesario dividir [0]/[1]
    self.edge_state[:][0] = link utilization 

    self.edge_state[:][1] = link capacity
    self.edge_state[:][2] = bw allocated (the one that goes from src to dst)
    """
    def __init__(self):
        self.graph = None # Here we store the graph as DiGraph (without repeated edges)
        self.source = None
        self.destination = None
        self.demand = None

        self.edge_state = None
        self.graph_topology_name = None # Here we store the name of the graph topology from the repetita dataset
        self.dataset_folder_name = None # Here we store the name of the repetita dataset being used: 2015Defo, 2016TopologyZoo_unary,2016TopologyZoo_inverseCapacity, etc. 

        self.diameter = None

        # Nx Graph where the nodes have features. Betweenness is allways normalized.
        # The other features are "raw" and are being normalized before prediction
        self.first = None #Tensor unidimensional con las posiciones de los enlaces mensajeros
        self.firstTrueSize = None
        self.second = None #Tensor unidimensional con las posiciones de los enlaces receptores
        self.between_feature = None

        self.percentage_demands = None # X% of the most loaded demands we use for optimization
        self.shufle_demands = False # If True we shuffle the list of traffic demands
        self.top_K_critical_demands = False # If we want to take the top X% of the 5 most loaded links
        self.num_critical_links = 5 #Los primeros X enlaces de mayor uso de los que tomaremos todas las demandas que pasen por ellos

        '''
        sp_middlepoints es un diccionario que contiene:
            * Clave: demanda, es decir --> origen:destino
            * Valor asociado: id del nodo de desvío
        Si un cierta demanda no existe en este diccionario, es porque no se está aplicando actualmente ningún desvío para dicha demanda.
        '''
        self.sp_middlepoints = None # For each src,dst we store the nodeId of the sp middlepoint
        self.shortest_paths = None # For each src,dst we store the shortest path to reach d
        self.sp_middlepoints_step = dict() # We store the midlepoint assignation before step() finishes

        # Mean and standard deviation of link betweenness
        self.mu_bet = None
        self.std_bet = None

        # Episode length in timesteps
        self.episode_length = None
        self.currentVal = None # Value used in hill_climbing way of choosing the next demand
        self.initial_maxLinkUti = None
        self.iter_list_elig_demn = None

        # Error at the end of episode to evaluate the learning process
        self.error_evaluation = None
        # Ideal target link capacity: self.sumTM/self.numEdges
        self.target_link_capacity = None

        self.TM = None # Traffic matrix where self.TM[src][dst] indicates how many packets are sent from src to dst
        self.sumTM = None
        self.routing = None # Loaded routing matrix
        self.paths_Matrix_from_routing = None # We store a list of paths extracted from the routing matrix for each src-dst pair

        self.K = None
        self.nodes = None # List of nodes to pick randomly from them
        self.ordered_edges = None
        #Diccionario que almacena, para cada enlace 'i:j', su posición en comparación al resto de enlaces
        self.edgesDict = dict() # Stores the position id of each edge in order
        self.previous_path = None

        self.src_dst_k_middlepoints = None # For each src, dst, we store the k middlepoints
        self.list_eligible_demands = None # Here we store those demands from DEFO that have one middlepoint. These demands are going to be eligible by our DRL agent.
        self.link_capacity_feature = None #Almacena la capacidad de cada enlace en relación a la capacidad del enlace de mayor capacidad

        self.numNodes = None
        self.numEdges = None
        self.next_state = None

        # We store the edge that has maximum utilization
        # (src, dst, MaxUtilization)
        self.edgeMaxUti = None 
        # We store the edge that has minimum utilization
        # (src, dst, MaxUtilization)
        self.edgeMinUti = None 
        # We store the path with more bandwidth from the edge with maximum utilization
        # (src, dst, MaxBandwidth)
        self.patMaxBandwth = None 
        self.maxBandwidth = None

        self.episode_over = True
        self.reward = 0
        self.allPaths = dict() # Stores the paths for each src:dst pair

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
    
    def add_features_to_edges(self):
        incId = 1
        for node in self.graph:
            #Recorremos las claves del diccionario asociado a dicho nodo 'node', con claves --> posiciones de los nodos vecinos
            for adj in self.graph[node]:
                #Trabajamos con un solo enlace dirigido de 'node' a 'adj', por lo que self.graph[node][adj] será un diccionario con una sola clave --> índice del único enlace dirigido (índice 0)
                if not 'betweenness' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['betweenness'] = 0
                if not 'edgeId' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['edgeId'] = incId
                if not 'numsp' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['numsp'] = 0
                if not 'utilization' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['utilization'] = 0
                if not 'capacity' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['capacity'] = 0
                if not 'weight' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['weight'] = 0
                if not 'kshortp' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['kshortp'] = 0
                if not 'crossing_paths' in self.graph[node][adj][0]: # We store all the src,dst from the paths crossing each edge
                    self.graph[node][adj][0]['crossing_paths'] = dict()
                incId = incId + 1

    def num_shortest_path(self, topology):
        self.diameter = nx.diameter(self.graph)
        # Iterate over all node1,node2 pairs from the graph
        for n1 in range (0,self.numNodes):
            for n2 in range (0,self.numNodes):
                if (n1 != n2):
                    # Check if we added the element of the matrix
                    if str(n1)+':'+str(n2) not in self.allPaths:
                        self.allPaths[str(n1)+':'+str(n2)] = []
                    # First we compute the shortest paths taking into account the diameter
                    [self.allPaths[str(n1)+':'+str(n2)].append(p) for p in nx.all_simple_paths(self.graph, source=n1, target=n2, cutoff=self.diameter*2)]

                    # We take all the paths from n1 to n2 and we order them according to the path length
                    # sorted() ordena los paths de menor a mayor numero de
                    # saltos y los que tienen los mismos saltos te los ordena por indice
                    self.allPaths[str(n1)+':'+str(n2)] = sorted(self.allPaths[str(n1)+':'+str(n2)], key=lambda item: (len(item), item))
                    path = 0
                    while path < self.K and path < len(self.allPaths[str(n1)+':'+str(n2)]):
                        currentPath = self.allPaths[str(n1)+':'+str(n2)][path]
                        i = 0
                        j = 1

                        # Iterate over pairs of nodes and allocate linkDemand
                        while (j < len(currentPath)):
                            self.graph.get_edge_data(currentPath[i], currentPath[j])[0]['numsp'] = \
                                self.graph.get_edge_data(currentPath[i], currentPath[j])[0]['numsp'] + 1
                            i = i + 1
                            j = j + 1

                        path = path + 1

                    # Remove paths not needed
                    del self.allPaths[str(n1)+':'+str(n2)][path:len(self.allPaths[str(n1)+':'+str(n2)])]
                    gc.collect()


    '''
    Este método será usado cada vez que querramos cambiar la ruta a seguir para llevar el tráfico desde init_source a final_destination, y suponiendo
    que la ruta actual ya está usando un middlepoint para seguir un desvío. En ese caso, tendremos que hacer lo siguiente:
        * Primero decrementar el uso de los enlaces que cruzamos para ir desde el nodo origen hasta el middlepoint
            Aquí src=nodo origen y dst=middlepoint
        * Segundo, decrementar el uso de los enlaces que definen la ruta a seguir para ir desde el middlepoint hasta el nodo final
            Aqúi src=middlepoint y dst=nodo final
    En resumen, nos permite eliminar un middlepoint usado para una determinada pareja de nodos.
    '''
    def decrease_links_utilization_sp(self, src, dst, init_source, final_destination):
        # In this function we desallocate the bandwidth by segments. This funcion is used when we want
        # to desallocate from a src to a middlepoint and then from middlepoint to a dst using the sp

        # We obtain the demand from the original source,destination pair
        bw_allocated = self.TM[init_source][final_destination]
        currentPath = self.shortest_paths[src,dst]

        i = 0
        j = 1
        while (j < len(currentPath)):
            firstNode = currentPath[i]
            secondNode = currentPath[j]

            self.graph[firstNode][secondNode][0]['utilization'] -= bw_allocated 
            if str(init_source)+':'+str(final_destination) in self.graph[firstNode][secondNode][0]['crossing_paths']:
                del self.graph[firstNode][secondNode][0]['crossing_paths'][str(init_source)+':'+str(final_destination)]
            self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][0] = self.graph[firstNode][secondNode][0]['utilization']
            i = i + 1
            j = j + 1

    '''
    Este método recibe como parámetro la lista de tuplas '(uso enlace, inicio enlace, final enlace)' o enlaces de mayor uso seleccioandos (los 'K' primeros).
    Las demandas críticas serán aquellas demandas que pasan por cualquiera de los K enlaces de mayor uso. Sobre esas demandas críticas, nos quedaremos con un porcentaje de ellas.
    Entonces, para cada enlace de mayor uso:
        * Tomamos las demandas que cruzan por dicho enlace
    Sobre el total de demandas críticas, ordenamos por ancho de banda y seleccionamos un porcentaje de ellas.
    
    El entorno posee un atributo 'list_eligible_demands' que almacena una lista de tuplas (origen,destino,ancho de banda). Esto es, guarda las demandas
    seleccionadas como críticas
    '''
    def _get_top_k_critical_flows(self, list_ids):
        self.list_eligible_demands.clear()
        for linkId in list_ids:
            #Para cada enlace (ancho de banda que transcurre por el, nodo origen, nodo destino)
            #Tomamos los nodos que están conectados mediante dicho enlace
            i = linkId[1] 
            j = linkId[2]
            #Recorremos las demandas que cruzan por dicho enlace
            #Accedemos al primer y único enlace dirigido i -> j, y tomamos la característica 'crossing_paths', que nos devuelve un diccionario de parejas (demanda (= origen:destino), tráfico a enviar)
            for demand, value in self.graph[i][j][0]['crossing_paths'].items():
                src, dst = int(demand.split(':')[0]), int(demand.split(':')[1])
                #Puede ser que esta demanda pase por otro de los enlaces seleccionados y ya se haya almacenado
                if (src, dst, self.TM[src,dst]) not in self.list_eligible_demands:  
                    self.list_eligible_demands.append((src, dst, self.TM[src,dst]))
        #Ya hemos tomado las demandas que cruzan pos los k enlaces de mayor uso. Ahora ordenamos por ancho de banda (de mayor a menor)
        self.list_eligible_demands = sorted(self.list_eligible_demands, key=lambda tup: tup[2], reverse=True)
        if len(self.list_eligible_demands)>int(np.ceil(self.numNodes*(self.numNodes-1)*self.percentage_demands)):
            #Seleccionamos un porcentaje de ellas
            self.list_eligible_demands = self.list_eligible_demands[:int(np.ceil(self.numNodes*(self.numNodes-1)*self.percentage_demands))]

    def _generate_tm(self, tm_id):
        # Sample a file randomly to initialize the tm
        graph_file = self.dataset_folder_name+"/"+self.graph_topology_name+".graph"
        # This 'results_file' file is ignored!
        results_file = self.dataset_folder_name+"/res_"+self.graph_topology_name+"_"+str(tm_id)
        tm_file = self.dataset_folder_name+"/TM/"+self.graph_topology_name+'.'+str(tm_id)+".demands"
        
        self.defoDatasetAPI = defoResults.Defo_results(graph_file,results_file)
        self.links_bw = self.defoDatasetAPI.links_bw
        self.MP_matrix = self.defoDatasetAPI.MP_matrix
        self.TM = self.defoDatasetAPI._get_traffic_matrix(tm_file)

        self.iter_list_elig_demn = 0
        self.list_eligible_demands.clear()
        min_links_bw = 1000000.0
        for src in range (0,self.numNodes):
            for dst in range (0,self.numNodes):
                if src!=dst:
                    #Guardamos las demandas de la matriz de tráfico candidatas a ser demandas críticas
                    self.list_eligible_demands.append((src, dst, self.TM[src,dst]))
                    # If we have a link between src and dst
                    if src in self.graph and dst in self.graph[src]:
                        # Store the link with minimum bw
                        if self.links_bw[src][dst]<min_links_bw:
                            min_links_bw = self.links_bw[src][dst]
                        
                        # Clear the link utilization and crossing paths for each link
                        self.graph[src][dst][0]['utilization'] = 0.0
                        self.graph[src][dst][0]['crossing_paths'].clear()
        
        # If we want to take the X% random demands
        if self.shufle_demands:
            random.shuffle(self.list_eligible_demands)
            self.list_eligible_demands = self.list_eligible_demands[:int(np.ceil(len(self.list_eligible_demands)*self.percentage_demands))]
        elif not self.top_K_critical_demands:
            # If we want to take the x% bigger demands
            #Ordenamos las demandas de la matriz de tráfico de mayor a menor tráfico
            self.list_eligible_demands = sorted(self.list_eligible_demands, key=lambda tup: tup[2], reverse=True)
            self.list_eligible_demands = self.list_eligible_demands[:int(np.ceil(len(self.list_eligible_demands)*self.percentage_demands))]

    def compute_link_utilization_reset(self):
        # Allocate for each src,dst the corresponding traffic on the corresponding SP
        for src in range (0,self.numNodes):
            for dst in range (0,self.numNodes):
                if src!=dst:
                    self.allocate_to_destination_sp(src, dst, src, dst)
    
    def _obtain_path_more_bandwidth_rand_link(self):
        # Obtain path with largest bandwidth from the edge with highest utilization
        # We sort the paths by bandwidth and pick random from the top 4
        sorted_dict = list((k, v) for k, v in sorted(self.graph[self.edgeMaxUti[0]][self.edgeMaxUti[1]][0]['crossing_paths'].items(), key=lambda item: item[1], reverse=True))
        path = random.randint(0, 1)
        # In case there is only one bandwidth
        if path>=len(sorted_dict):
            path = 0
        srcPath = int(sorted_dict[path][0].split(':')[0])
        dstPath = int(sorted_dict[path][0].split(':')[1])
        self.patMaxBandwth = (srcPath, dstPath, self.TM[srcPath][dstPath])
    
    def _obtain_path_from_set_rand(self):
        len_demans = len(self.list_eligible_demands)-1
        path = random.randint(0, len_demans)
        srcPath = int(self.list_eligible_demands[path][0])
        dstPath = int(self.list_eligible_demands[path][1])
        self.patMaxBandwth = (srcPath, dstPath, int(self.list_eligible_demands[path][2]))
    
    '''
    Este método nos permite seleccionar la siguiente demanda crítica que se devolverá con el objetivo de reenrutarla
    Tras invocar este método, se incrementa el iterador 'iter_list_elig_dem' de la lista, de forma que si volvemos a invocar a la función, se fijará la siguiente demanda crítica de la lista. 
    '''
    def _obtain_demand(self):
        src = self.list_eligible_demands[self.iter_list_elig_demn][0]
        dst = self.list_eligible_demands[self.iter_list_elig_demn][1]
        bw = self.list_eligible_demands[self.iter_list_elig_demn][2]
        self.patMaxBandwth = (src, dst, int(bw)) #Guardamos la siguiente demanda de mayor ancho de ancho de banda a reenrutar
        self.iter_list_elig_demn += 1
    
    def get_value(self, source, destination, action):
        # We get the K-middlepoints between source-destination
        middlePointList = self.src_dst_k_middlepoints[str(source) +':'+ str(destination)]
        middlePoint = middlePointList[action]

        # First we allocate until the middlepoint
        self.allocate_to_destination_sp(source, middlePoint, source, destination)
        # If we allocated to a middlepoint that is not the final destination
        if middlePoint!=destination:
            # Then we allocate from the middlepoint to the destination
            self.allocate_to_destination_sp(middlePoint, destination, source, destination)
            # We store that the pair source,destination has a middlepoint
            self.sp_middlepoints[str(source)+':'+str(destination)] = middlePoint
        
        currentValue = -1000000
        # Get the maximum loaded link and it's value after allocating to the corresponding middlepoint
        for i in self.graph:
            for j in self.graph[i]:
                position = self.edgesDict[str(i)+':'+str(j)]
                link_capacity = self.links_bw[i][j]
                if self.edge_state[position][0]/link_capacity>currentValue:
                    currentValue = self.edge_state[position][0]/link_capacity
        
        # Dissolve allocation step so that later we can try another action
        # Remove bandwidth allocated until the middlepoint and then from the middlepoint on
        if str(source)+':'+str(destination) in self.sp_middlepoints:
            middlepoint = self.sp_middlepoints[str(source)+':'+str(destination)]
            self.decrease_links_utilization_sp(source, middlepoint, source, destination)
            self.decrease_links_utilization_sp(middlepoint, destination, source, destination)
            del self.sp_middlepoints[str(source)+':'+str(destination)] 
        else: # Remove the bandwidth allocated from the src to the destination
            self.decrease_links_utilization_sp(source, destination, source, destination)
        
        return -currentValue  

    def _obtain_demand_hill_climbing(self):
        dem_iter = 0
        nextVal = -1000000
        self.next_state = None
        # Iterate for each demand possible
        for source in range(self.numNodes):
            for dest in range(self.numNodes):
                if source!=dest:
                    for action in range(len(self.src_dst_k_middlepoints[str(source)+':'+str(dest)])):
                        middlepoint = -1
                        # First we need to desallocate the current demand before we explore all it's possible actions
                        # Check if there is a middlepoint to desallocate from src-middlepoint-dst
                        if str(source)+':'+str(dest) in self.sp_middlepoints:
                            middlepoint = self.sp_middlepoints[str(source)+':'+str(dest)]
                            self.decrease_links_utilization_sp(source, middlepoint, source, dest)
                            self.decrease_links_utilization_sp(middlepoint, dest, source, dest)
                            del self.sp_middlepoints[str(source)+':'+str(dest)] 
                        else: # Remove the bandwidth allocated from the src to the destination
                            self.decrease_links_utilization_sp(source, dest, source, dest)

                        evalState = self.get_value(source, dest, action)
                        if evalState > nextVal:
                            nextVal = evalState
                            self.next_state = (action, source, dest)
                        
                        # Allocate back the demand whose actions we explored
                        # If the current demand had a middlepoint, we allocate src-middlepoint-dst
                        if middlepoint>=0:
                            # First we allocate until the middlepoint
                            self.allocate_to_destination_sp(source, middlepoint, source, dest)
                            # Then we allocate from the middlepoint to the destination
                            self.allocate_to_destination_sp(middlepoint, dest, source, dest)
                            # We store that the pair source,destination has a middlepoint
                            self.sp_middlepoints[str(source)+':'+str(dest)] = middlepoint
                        else:
                            # Then we allocate from the middlepoint to the destination
                            self.allocate_to_destination_sp(source, dest, source, dest)
        self.patMaxBandwth = (self.next_state[1], self.next_state[2], self.TM[self.next_state[1]][self.next_state[2]])

    def compute_middlepoint_set_random(self):
        # We choose the K-middlepoints for each src-dst randomly
        self.src_dst_k_middlepoints = dict()
        # Iterate over all node1,node2 pairs from the graph
        for n1 in range (0,self.numNodes):
            for n2 in range (0,self.numNodes):
                if (n1 != n2):
                    num_middlepoints = 0
                    self.src_dst_k_middlepoints[str(n1)+':'+str(n2)] = list()
                    # We add the destination as a candidate middlepoint (in case we have direct connection)
                    self.src_dst_k_middlepoints[str(n1)+':'+str(n2)].append(n2)
                    num_middlepoints += 1
                    while num_middlepoints<self.K:
                        middlpt = np.random.randint(0, self.numNodes)
                        while middlpt==n1 or middlpt==n2 or middlpt in self.src_dst_k_middlepoints[str(n1)+':'+str(n2)]:
                            middlpt = np.random.randint(0, self.numNodes)
                        self.src_dst_k_middlepoints[str(n1)+':'+str(n2)].append(middlpt)
                        num_middlepoints += 1         

        
    def mark_edges(self, action_flags, src, dst, init_source, final_destination):
        currentPath = self.shortest_paths[src,dst]
        
        i = 0
        j = 1

        while (j < len(currentPath)):
            firstNode = currentPath[i]
            secondNode = currentPath[j]

            action_flags[self.edgesDict[str(firstNode)+':'+str(secondNode)]] += 1.0
            i = i + 1
            j = j + 1
    
    def mark_action_to_edges(self, first_node, init_source, final_destination): 
        # In this function we mark for each link which is the bw that it will allocate. This we will
        # use to avoid repeated actions
        action_flags = np.zeros(self.numEdges)
        
        # Mark until first_node
        self.mark_edges(action_flags, init_source, first_node, init_source, final_destination)

        # If the first node is a middlepoint
        if first_node!=final_destination:
            self.mark_edges(action_flags, first_node, final_destination, init_source, final_destination)
        
        return action_flags

    def compute_middlepoint_set_remove_rep_actions_no_loop(self):
        # In this function we compute the middlepoint set but we don't take into account the middlepoints whose 
        # actions are repeated and neither those middlepoints whose SPs pass over the DST or SRC nodes
        
        # Compute SPs for each src,dst pair
        self.compute_SPs()

        # We compute the middlepoint set for each src,dst pair and we don't consider repeated actions
        self.src_dst_k_middlepoints = dict()
        # Iterate over all node1,node2 pairs from the graph
        for n1 in range (0,self.numNodes):
            for n2 in range (0,self.numNodes):
                if (n1 != n2):
                    self.src_dst_k_middlepoints[str(n1)+':'+str(n2)] = list()
                    repeated_actions = list()
                    for midd in range (0,self.K):
                        # If the middlepoint is not the source node
                        if midd!=n1:
                            action_flags = self.mark_action_to_edges(midd, n1, n2)
                            # If we allocated to a middlepoint that is not the final destination
                            if midd!=n2:
                                # If the repeated_actions list is empty we make the following verifications
                                if len(repeated_actions) == 0:
                                    #print(" A...... ")

                                    path1 = self.shortest_paths[n1, midd]
                                    path2 = self.shortest_paths[midd, n2]

                                    # Check that the dst node is not in the SP to avoid loops!
                                    currentPath = path1[:len(path1)-1]+path2
                                    dst_counter = 0
                                    for node in currentPath:
                                        if node==n2 or node==n1:
                                            dst_counter += 1
                                    # If there is only one dst node
                                    if dst_counter==2:
                                        repeated_actions.append(action_flags)
                                        self.src_dst_k_middlepoints[str(n1)+':'+str(n2)].append(midd)
                                else:
                                    #print(" B...... ")
                                    repeatedAction = False
                                    # Compare the current action with the previous ones
                                    for previous_actions in repeated_actions:
                                        subtraction = np.absolute(np.subtract(action_flags,previous_actions))
                                        if np.sum(subtraction)==0.0:
                                            repeatedAction = True
                                            break
                                    # If we didn't find any identical action, we make the following verifications
                                    if not repeatedAction:                                        
                                        path1 = self.shortest_paths[n1, midd]
                                        path2 = self.shortest_paths[midd, n2]
                                        # Check that the dst node is not in the SP to avoid loops!
                                        currentPath = path1[:len(path1)-1]+path2
                                        dst_counter = 0
                                        for node in currentPath:
                                            if node==n2 or node==n1:
                                                dst_counter += 1
                                        # If there is only one dst node
                                        if dst_counter==2:
                                            self.src_dst_k_middlepoints[str(n1)+':'+str(n2)].append(midd)
                                            repeated_actions.append(action_flags)

                            else: 
                                # If it's the first action we add it to the repeated actions list
                                if len(repeated_actions) == 0:
                                    #print(" C...... ")
                                    self.src_dst_k_middlepoints[str(n1)+':'+str(n2)].append(midd)
                                    repeated_actions.append(action_flags)
                                else:
                                    #print(" D...... ")
                                    repeatedAction = False
                                    # Compare the current action with the previous ones
                                    for previous_actions in repeated_actions:
                                        subtraction = np.absolute(np.subtract(action_flags,previous_actions))
                                        if np.sum(subtraction)==0.0:
                                            repeatedAction = True
                                            break
                                    
                                    # If we didn't find any identical action, we add the middlepoint to the set
                                    if not repeatedAction:
                                        self.src_dst_k_middlepoints[str(n1)+':'+str(n2)].append(midd)
                                        repeated_actions.append(action_flags)

    #Método que guarda la lista de nodos que conforman el camino más corto para cada pareja origen-destino, creando y guardando la info en shortest_paths.json
    def compute_SPs(self):
        diameter = nx.diameter(self.graph)
        self.shortest_paths = np.zeros((self.numNodes,self.numNodes),dtype=object)
        
        allPaths = dict()
        sp_path = self.dataset_folder_name+"/shortest_paths.json"

        if not os.path.isfile(sp_path):
            for n1 in range (0,self.numNodes):
                for n2 in range (0,self.numNodes):
                    if (n1 != n2):
                        allPaths[str(n1)+':'+str(n2)] = []
                        # First we compute the shortest paths taking into account the diameter
                        #all_simple_paths recibe el grafo de Networkx, y la fuente de origen y destino, devolviendo la lista de caminos que podemos seguir para ir de uno a otro
                        [allPaths[str(n1)+':'+str(n2)].append(p) for p in nx.all_simple_paths(self.graph, source=n1, target=n2, cutoff=diameter*2)]                    # We take all the paths from n1 to n2 and we order them according to the path length
                        # sorted() ordena los paths de menor a mayor numero de
                        # saltos y los que tienen los mismos saltos te los ordena por indice
                        aux_sorted_paths = sorted(allPaths[str(n1)+':'+str(n2)], key=lambda item: (len(item), item))                    # self.shortest_paths[n1,n2] = nx.shortest_path(self.graph, n1, n2,weight='weight')
                        allPaths[str(n1)+':'+str(n2)] = aux_sorted_paths[0]
        
            with open(sp_path, 'w') as fp:
                json.dump(allPaths, fp)
        else:
            allPaths = json.load(open(sp_path))

        for n1 in range (0,self.numNodes):
            for n2 in range (0,self.numNodes):
                if (n1 != n2):
                    self.shortest_paths[n1,n2] = allPaths[str(n1)+':'+str(n2)]
        
    '''
    Cada enlace dirigido del grafo (i:j) enviará mensajes a todos los enlaces dirigidos que salgan del nodo 'j' o que entren al nodo 'j'
    De esta forma: 
        * Los enlaces emisores de mensajes se almacenan en 'first'
        * Los enlaces receptores de mensajes se almacenan en 'second'
    '''
    def _first_second(self):
        first = list()
        second = list()

        for i in self.graph:
            for j in self.graph[i]:
                neighbour_edges = self.graph.edges(j) #Obtenemos todos los enlaces dirigidos que salen o entran al nodo 'j'
                # Take output links of node 'j'

                for m, n in neighbour_edges:
                    if ((i != m or j != n) and (i != n or j != m)):
                        first.append(self.edgesDict[str(i) +':'+ str(j)])
                        second.append(self.edgesDict[str(m) +':'+ str(n)])

        self.first = tf.convert_to_tensor(first, dtype=tf.int32)
        self.second = tf.convert_to_tensor(second, dtype=tf.int32)

    #Nos permite especificar la topología de trabajo y las matrices de tráfico con las que podemos trabajar para dicha topología 
    def generate_environment(self, dataset_folder_name, graph_topology_name, EPISODE_LENGTH, K, X):
        self.episode_length = EPISODE_LENGTH
        self.graph_topology_name = graph_topology_name
        self.dataset_folder_name = dataset_folder_name #Carpeta que contiene los datos de la topología (carpeta con matrices de tráfico, archivo con la especificación de la topología...)
        self.list_eligible_demands = list()
        self.iter_list_elig_demn = 0
        self.percentage_demands = X

        self.maxCapacity = 0 # We take the maximum capacity to normalize

        # Just select some random file, the only thing we need is the links features and the topology
        graph_file = self.dataset_folder_name+"/"+self.graph_topology_name+".graph"
        # This 'results_file' file is ignored!
        results_file = self.dataset_folder_name+"/res_"+self.graph_topology_name+"_0"
        tm_file = self.dataset_folder_name+"/TM/"+self.graph_topology_name+".0.demands"
        self.defoDatasetAPI = defoResults.Defo_results(graph_file,results_file)
        
        self.graph = self.defoDatasetAPI.Gbase
        #Añadimos características a los enlaces insertados en el grafo dirigido
        #Recordemos que podemos añadir más de un enlace dirigido del nodo A al nodo B. Por tanto, cuando creamos un primero, se identifica con la clave 0
        #De esta forma, grafo[A][B] es un diccionario con claves 0,1 .... que identifican a los enlaces dirigidos que van de A a B, y valores las características de los nodos con claves 0,1...
        self.add_features_to_edges() #Construye las características asociadas a cada enlace dirigido, usando como valor inicial 0
        self.numNodes = len(self.graph.nodes())
        self.numEdges = len(self.graph.edges())
        btwns = nx.edge_betweenness_centrality(self.graph)

        self.K = K
        if self.K>self.numNodes:
            self.K = self.numNodes

        #Mantenemos un array 2D de numpy (matriz) con 3 columnas para cada enlace dirigido
        self.edge_state = np.zeros((self.numEdges, 3))
        self.betweenness_centrality = np.zeros(self.numEdges) # Used in the fully connected
        self.shortest_paths = np.zeros((self.numNodes,self.numNodes),dtype="object")

        position = 0
        #Procedemos a rellenar con los valores tomados del .graph e insertados en el grafo de Networkx las características previamente creados usando add_features_to_edges() 
        for i in self.graph:
            for j in self.graph[i]:
                self.edgesDict[str(i)+':'+str(j)] = position
                self.graph[i][j][0]['capacity'] = self.defoDatasetAPI.links_bw[i][j]
                self.graph[i][j][0]['weight'] = self.defoDatasetAPI.links_weight[i][j]
                if self.graph[i][j][0]['capacity']>self.maxCapacity:
                    self.maxCapacity = self.graph[i][j][0]['capacity']
                self.edge_state[position][1] = self.graph[i][j][0]['capacity'] #Capacidad del único enlace dirigido que va del nodo 'i' al nodo 'j'
                #self.betweenness_centrality[position] = btwns[i,j]
                self.betweenness_centrality[position] = btwns.get((i, j), btwns.get((j, i), 0.0))  # Robusto
                self.graph[i][j][0]['utilization'] = 0.0
                self.graph[i][j][0]['crossing_paths'].clear()
                position += 1

        #Guardamos los enlaces emisores y receptores de mensajes
        self._first_second()
        self.firstTrueSize = len(self.first)

        self.link_capacity_feature = tf.convert_to_tensor(np.divide(self.edge_state[:,1], self.maxCapacity), dtype=tf.float32)
        self.betweenness_centrality = tf.convert_to_tensor(self.betweenness_centrality, dtype=tf.float32)

        # We create the list of nodes ids to pick randomly from them
        self.nodes = list(range(0,self.numNodes))

        self.compute_middlepoint_set_remove_rep_actions_no_loop()

    #Método encargado de ejecutar una acción en el entorno actual
    def step(self, action, demand, source, destination):
        # Action is the middlepoint. Careful because it can also be action==destination if src,dst are connected directly by an edge
        self.episode_over = False
        self.reward = 0

        # We get the K-middlepoints between source-destination
        #Obtenemos la lista de middepoints para dicha pareja origen-destino, y tomamos el middlepoint especificado en 'action'
        middlePointList = self.src_dst_k_middlepoints[str(source) +':'+ str(destination)]
        middlePoint = middlePointList[action]

        # First we allocate until the middlepoint
        #Primero incrementamos el uso de los enlaces que hay que cruzar para ir desde origen hasta middlepoint
        self.allocate_to_destination_sp(source, middlePoint, source, destination)
        # If we allocated to a middlepoint that is not the final destination
        #Puede ser que el middlepoint escogido sea el propio nodo destino, en caso de que el nodo origen esté relacionado directamente con un enlace al nodo destino. Lo comprobamos
        if middlePoint!=destination:
            #Si el middlepoint seleccionado no es el propio nodo destino, hay que plasmar la segunda parte de la ruta: middlepoint - destino
            # Then we allocate from the middlepoint to the destination
            self.allocate_to_destination_sp(middlePoint, destination, source, destination)
            # We store that the pair source,destination has a middlepoint
            #Guardamos el middlepoint usado para dicha pareja origen-destino
            self.sp_middlepoints[str(source)+':'+str(destination)] = middlePoint
        
        self.sp_middlepoints_step = self.sp_middlepoints
        
        # Find new maximum and minimum utilization link
        '''
        Ahora debemos actualizar el nuevo enlace de mayor uso. Para ello, se siguen varios pasos:
            * Se guarda el uso del enlace de mayor uso actual, antes de actualizar el nuevo enlace de mayor uso.
            * Actualizamos el enlace de mayor uso modelado mediante la variable 'edgeMaxUti' : (inicio,fin,uso)
                - inicio y fin reflejan los extremos del enlace dirigido de mayor uso
                - 'uso' refleja la utilización de dicho enlace modelado como inicio-fin (enlace dirigido que va de 'inicio' a 'fin')
        '''
        old_Utilization = self.edgeMaxUti[2]
        self.edgeMaxUti = (0, 0, 0)
        for i in self.graph:
            for j in self.graph[i]:
                position = self.edgesDict[str(i)+':'+str(j)]
                self.edge_state[position][0] = self.graph[i][j][0]['utilization']
                link_capacity = self.links_bw[i][j]
                norm_edge_state_capacity = self.edge_state[position][0]/link_capacity
                if norm_edge_state_capacity>self.edgeMaxUti[2]:
                    self.edgeMaxUti = (i, j, norm_edge_state_capacity) #Se guarda una tupla con el enlace de mayor uso en la red, y su uso
         
        self.currentVal = -self.edgeMaxUti[2]
        #Calculamos la recompensa de llevar a cabo dicha acción sobre dicho estado (diferencia de uso de los enlaces de mayor uso atiguo y nuevo)
        self.reward = np.around((old_Utilization-self.edgeMaxUti[2])*10,2)

        # If we didn't iterate over all demands
        #Si quedan demandas críticas por reenrutar, configuramos el entorno para que seleccione internamente la siguiente.
        if self.iter_list_elig_demn<len(self.list_eligible_demands):
            self._obtain_demand()
        else:
            src = 1
            dst = 2
            bw = self.TM[src][dst]
            self.patMaxBandwth = (src, dst, int(bw))
            self.episode_over = True

        # Remove bandwidth allocated until the middlepoint and then from the middlepoint on
        '''
        sp_middlepoints es un diccionario [origen:destino] (clave) -> [id_middlepoint usado] (valor)
        Tras asignar un middlepoint a la demanda correspondiente, se procederá a eliminar la ruta actual que sigue patMaxBandwth, que representa
        la siguiente demanda crítica a tratar (fijada en el  método previo _obatin_demand).
        '''
        if str(self.patMaxBandwth[0])+':'+str(self.patMaxBandwth[1]) in self.sp_middlepoints:
            #La siguiente demanda crítica a reenrutar ya tiene asignado un middlepoint
            middlepoint = self.sp_middlepoints[str(self.patMaxBandwth[0])+':'+str(self.patMaxBandwth[1])]
            self.decrease_links_utilization_sp(self.patMaxBandwth[0], middlepoint, self.patMaxBandwth[0], self.patMaxBandwth[1])
            self.decrease_links_utilization_sp(middlepoint, self.patMaxBandwth[1], self.patMaxBandwth[0], self.patMaxBandwth[1])
            del self.sp_middlepoints[str(self.patMaxBandwth[0])+':'+str(self.patMaxBandwth[1])] 
        else: # Remove the bandwidth allocated from the src to the destination
            #No tiene asignado un middlepoint, por lo que la ruta es la más corta que va de origen a destino
            self.decrease_links_utilization_sp(self.patMaxBandwth[0], self.patMaxBandwth[1], self.patMaxBandwth[0], self.patMaxBandwth[1])
        
        # We desmark the bw_allocated
        self.edge_state[:,2] = 0

        '''
        Devolvemos, cada vez que el agente aplica una acción sobre el entorno (reenruta una cierta demanda):
            * Recompensa obtenida (diferencia en el uso de los enlaces de mayor uso antiguo y nuevo)
            * Si hemos terminado o no de reenrutar todas las demandas críticas seleccionadas
            * Siempre que tras aplicar la acción, siga quedando al menos una demanda más que reenrutar, devuelve (ancho de banda, origen, destino) de la siguiente demanda a reenrutar
            * Uso del enlace de mayor uso actual tras aplicar la acción
        '''
        return self.reward, self.episode_over, 0.0, self.TM[self.patMaxBandwth[0]][self.patMaxBandwth[1]], self.patMaxBandwth[0], self.patMaxBandwth[1], self.edgeMaxUti, 0.0, np.std(self.edge_state[:,0])

    '''
    Método encargado de reiniciar el entorno:
        * Se resetea el entorno, usando una matriz de tráfico diferente (es decir, cambiando el ancho de banda a envíar) pero manteniendo el msimo enrutamiento
        * Se devuelve la demanda crítica de mayor ancho de banda que es necesaria reenrutar, eliminando claramente el ancho de banda de dicha demanda
          de los enlaces que constituyen el camino que seguía el tráfico asociado a esa demanda.
    '''
    def reset(self, tm_id):
        """
        Reset environment and setup for new episode. 
        Generate new TM but load the same routing. We remove the path with more bandwidth
        from the link with more utilization to later allocate it on a new path in the act().
        """
        #Cambiamos la matriz de tráfico, esto es, el ancho de banda de las diferentes demandas de tráfico
        #Este método guarda las demandas enviadas para cada pareja de nodo en dicha matriz, y guarda las demandas de mayor ancho de banda
        self._generate_tm(tm_id)

        self.sp_middlepoints = dict()

        # For each link we store the total sum of bandwidths of the paths crossing each link without middlepoints
        #Se vuelve a restablecer el ancho de banda de los enlaces, suponiendo de nuevo que la ruta inicial de cada demanda es la definida por OSPF
        self.compute_link_utilization_reset()

        # We iterate over all links in an ordered fashion and store the features to edge_state
        self.edgeMaxUti = (0, 0, 0)
        # This list is used to obtain the top K flows from the critical links
        #Esta lista contiene tuplas con la info de todos y cada uno de los enlaces de la topología
        list_link_uti_id = list()
        for i in self.graph:
            for j in self.graph[i]:
                position = self.edgesDict[str(i)+':'+str(j)]
                self.edge_state[position][0] = self.graph[i][j][0]['utilization']
                self.edge_state[position][1] = self.graph[i][j][0]['capacity']
                link_capacity = self.links_bw[i][j]
                # We store the link utilization and the corresponding edge
                #Guardamos tuplas de la forma (ancho de banda que transcurre por el enlace i-j, i, j)
                list_link_uti_id.append((self.edge_state[position][0], i, j))
                
                norm_edge_state_capacity = self.edge_state[position][0]/link_capacity #Uso del enlace actual
                if norm_edge_state_capacity>self.edgeMaxUti[2]: #Si el uso de dicho enlace es superior al uso del enlace de mayor uso...
                    self.edgeMaxUti = (i, j, norm_edge_state_capacity)
        
        #Esto está inicialmente desactivado. Pero cuando instancie el entorno en train_Enero_3_top_script.py, lo activo --> para tomar así las demandas que pasan por los 5 enlaces de mayor uso
        if self.top_K_critical_demands:
            #Tomamos los primeros K enlaces más congestionados (en cuanto a su uso)
            list_link_uti_id = sorted(list_link_uti_id, key=lambda tup: tup[0], reverse=True)[:self.num_critical_links]
            #Pasamos al entorno la lista de los 'K' enlaces más congestionados.
            self._get_top_k_critical_flows(list_link_uti_id)

        self.currentVal = -self.edgeMaxUti[2]
        self.initial_maxLinkUti = -self.edgeMaxUti[2]
        # From the link with more utilization, we obtain a random path of the 5 with more bandwidth
        #self._obtain_path_more_bandwidth_rand_link()
        #self._obtain_path_from_set_rand()
        #self._obtain_demand_hill_climbing()
        
        #Fijamos en el entorno la siguiente demanda crítica de la lista de demandas críticas 'list_eligible_demands', almacenándose en 'patMaxBanwth': tupla (origen,destino,ancho de banda)
        self._obtain_demand()

        # Remove bandwidth allocated for the path with more bandwidth from the link with more utilization
        #Como este método devuelve la siguiente demanda crítica a reenrutar, será necesario primero eliminar la ruta actual, desalojando el ancho de banda de los enlaces
        #por lo que cruce la ruta actual.
        self.decrease_links_utilization_sp(self.patMaxBandwth[0], self.patMaxBandwth[1], self.patMaxBandwth[0], self.patMaxBandwth[1])

        # We desmark the bw_allocated
        self.edge_state[:,2] = 0

        return self.TM[self.patMaxBandwth[0]][self.patMaxBandwth[1]], self.patMaxBandwth[0], self.patMaxBandwth[1]

    '''
    A diferencia del método 'mark_action_sp', este método nos permite aplicar realmente sobre el grafo un desvío a seguir, actualizando directamente el uso
    de los enlaces de la ruta definida.
    A la hora de decidir realizar un cierto desvío al enviar los datos origen-destino, este método debe ser invocado dos veces:
        * Una primera vez para actualizar los enlaces del grafo que definen la ruta 'origen-middlepoint'
        * Una segunda vez para lo mismo, en este caso, trabajando sobre los enlaces del camino que nos lleva del middlepoint al destino
    '''
    def allocate_to_destination_sp(self, src, dst, init_source, final_destination): 
        # In this function we allocated the bandwidth by segments. This funcion is used when we want
        # to allocate from a src to a middlepoint and then from middlepoint to a dst using the sp
        bw_allocate = self.TM[init_source][final_destination]
        #camino más corto para ir desde origen a 'dst', que será el nodo de desvío : lista de nodos a seguir
        currentPath = self.shortest_paths[src,dst]
        
        i = 0
        j = 1
        #Actualizamos el uso de los enlaces que hay que cruzar en el camino src-dst (origen-middlepoint o middlepoint-destino)
        while (j < len(currentPath)):
            firstNode = currentPath[i]
            secondNode = currentPath[j]

            #Actualizamos el objeto grafo en sí
            self.graph[firstNode][secondNode][0]['utilization'] += bw_allocate  
            self.graph[firstNode][secondNode][0]['crossing_paths'][str(init_source)+':'+str(final_destination)] = bw_allocate
            #Actualizamos tb el vector de estados de los enlaces del grafo, que mantiene aparte el entorno
            self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][0] = self.graph[firstNode][secondNode][0]['utilization']
            i = i + 1
            j = j + 1
    
    '''
    Este método marca los enlaces que tenemos que cruzar si queremos realizar un desvío al envíar los datos de origen a destino.
    Para marcar el camino a seguir, se seguirán los siguientes pasos:
        * Se seleccionan los enlaces del camino a seguir
        * Para cada uno de esos enlaces, se toma su vector de estado, y se asigna en la componente '2': uso del enlace considerando únicamente el ancho de banda de
        la demanda que queremos enrutar realizando un cierto desvío. Esto nos permite tener en todos aquellos enlaces que definen el camino a seguir, cuanta de su capacidad
        necesitamos para transferir ese ancho de banda.

        Gracias a esto, podemos marcar el camino de forma "abstracta", sin toquetear el estado actual de los enlaces del grafo, es decir, sin plasmar realmente el camino.
        Por esta razón, se evita tocar directamente el grafo en sí, toqueteando únicamente los vectores estado que mantiene el agente para cada enlace.
    '''
    def mark_action_sp(self, src, dst, init_source, final_destination): 
        # In this function we mark the action in the corresponding edges of the SP between src,dst

        #Tomamos el ancho de banda a enviar de origen a destino
        bw_allocate = self.TM[init_source][final_destination]

        #Tomamos el camino mas corto para ir desde el nodo origen hasta el nodo intermedio de desvío 'dst'
        #Esto devuelve una lista con el índice de los nodos del camino más corto
        currentPath = self.shortest_paths[src,dst] 
        
        i = 0
        j = 1

        while (j < len(currentPath)):
            firstNode = currentPath[i]
            secondNode = currentPath[j]
            #Almacenamos en cada uno de los enlaces a recorrer el uso pero considerando SOLO el ancho de banda de la demanda que queremos enrutar en este preciso momento.
            #Realmente se marcan los enlaces para seguir ese desvío, pero indicando el uso del enlace considerando únicamente dicho desvío
            self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][2] = bw_allocate/self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][1]
            i = i + 1
            j = j + 1