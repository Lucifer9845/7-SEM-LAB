graph = {
    'A': [[('B', 1), ('C', 1)], [('D', 1)]],
    'B': [[('G', 1)], [('H', 1)]],
    'C': [[('J', 1)]],
    'D': [[('E', 1), ('F', 1)]],
    'G': [[('I', 1)]]
}

H = {
    'A': 1, 
    'B': 6,
    'C': 2,
    'D': 12,
    'E': 2,
    'F': 1,
    'G': 5,
    'H': 7,
    'I': 7,
    'J': 1,
    'T': 3
}

start = 'A'

parent = {}
status = {}
solution = {}

def computeMinimumCostChildNodes(currNode): 
    minimumCost = 0
    
    costToChildNodeListDict = {}
    costToChildNodeListDict[minimumCost] = []
    
    firstIteration = True
    
    for nextNodeTupleArray in graph.get(currNode,''): 
        currNodeCost = 0
        nextNodeArray = []
        for node, weight in nextNodeTupleArray:
            currNodeCost += H.get(node, 0) + weight
            nextNodeArray.append(node)

        if firstIteration == True: 
            minimumCost = currNodeCost
            costToChildNodeListDict[minimumCost] = nextNodeArray 
            firstIteration = True
        else: 
            if currNodeCost < minimumCost:
                minimumCost = currNodeCost
                costToChildNodeListDict[minimumCost] = nextNodeArray 


    return minimumCost, costToChildNodeListDict[minimumCost] 

def aoStar(v, backTracking):
    print("PROCESSING NODE :", v)
    print("HEURISTIC VALUES :", H)
    print("SOLUTION GRAPH :", solution)
    print()
    
    # if status is non negative
    if status.get(v, 0) >= 0: 
        # 1 - calc minimum cost, child node list
        minimumCost, childNodeList = computeMinimumCostChildNodes(v)
        H[v] = minimumCost
        status[v] = len(childNodeList)

        # 2 - set parents
        for childNode in childNodeList:
            parent[childNode] = v

        # 3 - eval solved, if any child is not solved. set solved = false
        solved = True 
        for childNode in childNodeList:
            if status.get(childNode, 0) != -1:
                solved = False

        # 4 - if solved set status to -1, solution to child node list
        if solved == True:
            status[v] = -1 
            solution[v] = childNodeList 

        # 5 - if not start node then recurse without backtracking
        if v != start: 
            aoStar(parent[v], False) 

        # 6 - if backtrack, apply algo for all children
        if backTracking == True: 
            for childNode in childNodeList: 
                aoStar(childNode, True) 
    
    
aoStar(start, True)

print("\nSOLUTION :", solution)