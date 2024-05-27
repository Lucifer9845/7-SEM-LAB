# 1.) Implement A* Search algorithm.

graph = {  # graph with edges
    'A': [('B', 6), ('F', 3)],
    'B': [('D', 2), ('C', 3)],
    'C': [('D', 1), ('E', 5)],
    'D': [('C', 1), ('E', 8)],
    'E': [('I', 5), ('J', 5)],
    'F': [('G', 1), ('H', 7)],
    'G': [('I', 3)],
    'H': [('I', 2)],
    'I': [('E', 5), ('J', 3)],
}

h_cost = {  # h cost of all
    'A': 10,
    'B': 8,
    'C': 5,
    'D': 7,
    'E': 3,
    'F': 6,
    'G': 5,
    'H': 3,
    'I': 1,
    'J': 0
}


def nextNodes(v):
    if v in graph:
        return graph[v]
    else:
        return None


def astarAlgo(start_node, stop_node):
    OPEN = set(start_node)  # put start in OPEN
    CLOSE = set()  # CLOSE is empty
    g_cost = {}  # dictionary to store g_costs
    parents = {}  # dictionary of parent nodes of final path

    g_cost[start_node] = 0  # g_cost of start node is 0
    parents[start_node] = start_node

    while len(OPEN) > 0:
        node = None

        # find node with min cost(g+h)
        for currNode in OPEN:
            if node == None or g_cost[currNode] + h_cost[currNode] < g_cost[node] + h_cost[node]:
                node = currNode

        # if node is not last or edges are connected with this node then pass
        if node != stop_node and graph[node] != None:
            
            # for each next node
            for (nextNode, weight) in nextNodes(node):
                # if not yet visited
                if nextNode not in OPEN and nextNode not in CLOSE:
                    OPEN.add(nextNode)
                    parents[nextNode] = node
                    g_cost[nextNode] = g_cost[node] + weight
                    
                else:
                    # if visited then set cost to min
                    if g_cost[node] + weight < g_cost[nextNode]:
                        g_cost[nextNode] = g_cost[node] + weight
                        parents[nextNode] = node
                        if nextNode in CLOSE:  #add in open if already in close
                            CLOSE.remove(nextNode)
                            OPEN.add(nextNode)
                            
        if node == None:
            print("Path does not exist")
            return None

        if node == stop_node:
            path = []
            
            while parents[node] != node:
                path.append(node)
                node = parents[node]
                
            path.append(start_node)
            path.reverse()
            print("Path found {}".format(path))
            return path

        OPEN.remove(node)
        CLOSE.add(node)
    print("Path does not exist")
    return None


astarAlgo('A', 'J')
