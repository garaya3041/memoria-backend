import math, time
import numpy as np
from scipy.optimize import linprog
from random import randrange

MAXITER = 100000

class Node:
    def __init__(self, dictionary):
        self.left = Node(dictionary["left"]) if ("left" in dictionary) and dictionary["left"] is not None else None 
        self.right = Node(dictionary["right"]) if ("right" in dictionary) and dictionary["right"] is not None else None
        self.A_ub = dictionary["A_ub"] if ("A_ub" in dictionary) else None
        self.B_ub = dictionary["B_ub"] if ("B_ub" in dictionary) else None
        self.A_eq = dictionary["A_eq"] if ("A_eq" in dictionary) else None
        self.B_eq = dictionary["B_eq"] if ("B_eq" in dictionary) else None
        self.C = dictionary["C"] if ("C" in dictionary) else None
        self.z = dictionary["z"] if ("z" in dictionary) else None
        self.x = dictionary["x"] if ("x" in dictionary) else None
        self.feasible = dictionary["feasible"] if ("feasible" in dictionary) else False
        self.mark = dictionary["mark"] if ("mark" in dictionary) else "active"
        self.id = dictionary["id"] if ("id" in dictionary) else 1
        self.level = dictionary["level"] if ("level" in dictionary) else 0

    def solve(self):
        solution = linprog(
            c = np.array(self.C),
            A_ub = np.array(self.A_ub) if self.A_ub else None,
            b_ub = np.array(self.B_ub) if self.B_ub else None,
            A_eq = np.array(self.A_eq) if self.A_eq else None,
            b_eq = np.array(self.B_eq) if self.B_eq else None, 
            method = 'revised simplex'
        )
        if solution.success:
            self.z = solution.fun.tolist()
            self.x = solution.x.tolist()
            self.feasible = True
        else:
            self.mark = "infeasible"
            print(f"{self.id} was marked as infeasible")
    
    def areIntegers(self):
        bool = True
        for i in self.x:
            bool = bool and not i%1
        return bool

    def toDictionary(self):
        obj = {
            "left": self.left.toDictionary() if self.left else None,
            "right": self.right.toDictionary() if self.right else None,
            "A_ub": self.A_ub,
            "B_ub": self.B_ub,
            "A_eq": self.A_eq,
            "B_eq": self.B_eq,
            "C": self.C,
            "z": self.z,
            "x": self.x,
            "feasible": self.feasible,
            "mark": self.mark,
            "id": self.id,
            "level": self.level
        }
        return obj
    
    def getIndex(self, method="more-restricted"):
        if method == "random":
            return self.random()
        if method == "lexic":
            return self.lexicOrder("right")
        if method == "inverse-lexic":
            return self.lexicOrder("inverse")
        if method == "nearest-toInt":
            return self.nearToInt("most")
        if method == "farthest-toInt":
            return self.nearToInt("least")
        if method == "maxValue":
            return self.inValue("most")
        if method == "minValue":
            return self.inValue("least")
        if method == "more-restricted":
            return self.restricted("more")
        if method == "less-restricted":
            return self.restricted("less")
        if method == "more-connected":
            return self.connected("more")
        if method == "less-connected":
            return self.connected("less")

    def random(self):
        x = np.array(self.x)
        x = np.round(x,6)
        x = x%1!=0
        nonIntegers = []
        for idx, val in enumerate(x):
            if val:
                nonIntegers.append(idx)
        if len(nonIntegers) > 0:
            idx = np.random.choice(nonIntegers).tolist()
            return idx
        else:
            return False

    def lexicOrder(self, method="left-right"):
        x = np.array(self.x)
        x = np.round(x,6)
        x = x%1!=0
        nonIntegers = []
        for idx, val in enumerate(x):
            if val:
                nonIntegers.append(idx)
        if len(nonIntegers)>0:
            if method=="right":
                return nonIntegers[0]
            elif method=="inverse":
                return nonIntegers[-1]
            else:
                return False
        else:
            return False
    
    def nearToInt(self, method="most"):
        x = np.array(self.x)
        x = np.round(x,6)
        absX = np.absolute(x)
        normalization = absX - np.floor(absX)
        maxi = float('-inf')
        maxIdx = None
        mini = float('inf')
        minIdx = None
        for idx,val in enumerate(normalization):
            if val <= 0.5 and val != 0:
                if val > maxi:
                    maxi = val
                    maxIdx = idx
                if val < mini:
                    mini = val
                    minIdx = idx
            if val > 0.5:
                if 1-val > maxi:
                    maxi = 1-val
                    maxIdx = idx
                if val < mini:
                    mini = 1-val
                    minIdx = idx
        if maxIdx is not None and minIdx is not None:
            if method == "most":
                return maxIdx
            elif method == "least":
                return minIdx
            else:
                return False
        else:
            return False

    def inValue(self, method="most"):
        x = np.array(self.x)
        x = np.round(x,6)
        x_bool = x%1!=0
        maxi = float('-inf')
        maxIdx = None
        mini = float('inf')
        minIdx = None
        for idx,val in enumerate(x_bool):
            if val:
                if x[idx] > maxi:
                    maxi = x[idx]
                    maxIdx = idx
                if x[idx] < mini:
                    mini = x[idx]
                    minIdx = idx
        if maxIdx is not None and minIdx is not None:
            if method == "most":
                return maxIdx
            elif method == "least":
                return minIdx
            else:
                return False
        else:
            return False

    def connected(self, method="more"):
        x = np.array(self.x)
        x = np.round(x,6)
        x = x%1!=0
        a1 = None
        if self.A_ub and self.A_eq:
            a = np.array(self.A_ub) !=0
            b = np.array(self.A_eq) !=0
            a1 = np.concatenate([a,b])
        elif self.A_ub:
            a1 = np.array(self.A_ub) !=0
        elif self.A_eq:
            a1 = np.array(self.A_eq) !=0
        else:
            return False
        a1 = a1.transpose().dot(a1)
        ones = np.ones(a1.shape[0])
        restrictions = ones.dot(a1) - 1
        totalConnections = []
        if method=="more":
            for idx, val in enumerate(x):
                if val:
                    totalConnections.append(restrictions[idx])
                else:
                    totalConnections.append(0)
            totalConnections = np.array(totalConnections)
            return int(np.argmax(totalConnections))
        elif method=="less":
            for idx, val in enumerate(x):
                if val:
                    totalConnections.append(restrictions[idx])
                else:
                    totalConnections.append(float('inf'))
            totalConnections = np.array(totalConnections)
            return int(np.argmin(totalConnections))
        else:
            return False
    
    def restricted(self, method="more"):
        x = np.array(self.x)
        x = np.round(x,6)
        x = x%1!=0
        a1 = None
        if self.A_ub and self.A_eq:
            a = np.array(self.A_ub) !=0
            b = np.array(self.A_eq) !=0
            a1 = np.concatenate([a,b])
        elif self.A_ub:
            a1 = np.array(self.A_ub) !=0
        elif self.A_eq:
            a1 = np.array(self.A_eq) !=0
        else:
            return False
        ones = np.ones(a1.shape[0])
        restrictions = ones.dot(a1)
        totalRestrictions = []
        if method=="more":
            for idx, val in enumerate(x):
                if val:
                    totalRestrictions.append(restrictions[idx])
                else:
                    totalRestrictions.append(0)
            totalRestrictions = np.array(totalRestrictions)
            return int(np.argmax(totalRestrictions))
        elif method=="less":
            for idx, val in enumerate(x):
                if val:
                    totalRestrictions.append(restrictions[idx])
                else:
                    totalRestrictions.append(float('inf'))
            totalRestrictions = np.array(totalRestrictions)
            return int(np.argmin(totalRestrictions))
        else:
            return False

    def branching(self, maxId, method="random"):
        varIdx = self.getIndex(method)
        if varIdx is not False:
            newValue = math.floor(self.x[varIdx])
            newConstraint_left = np.zeros(np.array(self.x).shape[0])
            newConstraint_right = np.zeros(np.array(self.x).shape[0])
            newConstraint_left[varIdx]=1
            newConstraint_right[varIdx]=-1
            a = np.array(self.A_ub)
            b = np.array(self.B_ub)
            a_left = np.concatenate([a,[newConstraint_left]])
            a_right = np.concatenate([a,[newConstraint_right]])
            b_left = np.concatenate([b,[newValue]])
            b_right = np.concatenate([b,[-newValue-1]])
            level = self.level + 1
            obj_left = {
                "A_ub": a_left.tolist(),
                "B_ub": b_left.tolist(),
                "A_eq": self.A_eq,
                "B_eq": self.B_eq,
                "C": self.C,
                "id": maxId+1,
                "level": level
            }
            obj_right = {
                "A_ub": a_right.tolist(),
                "B_ub": b_right.tolist(),
                "A_eq": self.A_eq,
                "B_eq": self.B_eq,
                "C": self.C,
                "id": maxId+2,
                "level": level
            }
            newNode_left = Node(obj_left)
            newNode_left.solve()
            newNode_right = Node(obj_right)
            newNode_right.solve()
            self.left = newNode_left
            self.right = newNode_right
            dictionary = {
                "leftZ": self.left.z,
                "leftX": self.left.x,
                "leftId": self.left.id,
                "leftFeasible": self.left.feasible,
                "leftVarIdx": varIdx,
                "leftNewValue": newValue,
                "rightZ": self.right.z,
                "rightX": self.right.x,
                "rightId": self.right.id,
                "rightFeasible": self.right.feasible,
                "rightVarIdx": varIdx,
                "rightNewValue": newValue+1,
                "height": level
            }
            return dictionary
        else:
            return False
    

class Tree:
    def __init__(self, treeDictionary = None):
        if "root" in treeDictionary:
            self.root = Node(treeDictionary["root"])
        else:
            self.root = Node(treeDictionary) if (treeDictionary) else None
        if(treeDictionary):
            self.firstIteration = treeDictionary["firstIteration"] if ("firstIteration" in treeDictionary) else True
            self.maxId = treeDictionary["maxId"] if ("maxId" in treeDictionary) else 1
            self.activeNodesIds = treeDictionary["activeNodesIds"] if ("activeNodesIds" in treeDictionary) else None
            self.bestZ = treeDictionary["bestZ"] if ("bestZ" in treeDictionary) else None
            self.bestX = treeDictionary["bestX"] if ("bestX" in treeDictionary) else None
            self.bestIter = treeDictionary["bestIter"] if ("bestIter" in treeDictionary) else None
            self.bestTime = treeDictionary["bestTime"] if ("bestTime" in treeDictionary) else None
            self.firstIntegerIter = treeDictionary["firstIntegerIter"] if ("firstIntegerIter" in treeDictionary) else None
            self.firstIntegerTime = treeDictionary["firstIntegerTime"] if ("firstIntegerTime" in treeDictionary) else None
            self.type = treeDictionary["type"] if ("type" in treeDictionary) else "min"
            self.typeSearch = treeDictionary["typeSearch"] if ("typeSearch" in treeDictionary) else "inOrder"
            self.method = treeDictionary["method"] if ("method" in treeDictionary) else "random"
            self.iterator = treeDictionary["iterator"] if ("iterator" in treeDictionary) else {
                "type": "unlimited",
                "iteration-steps": 0,
                "time-limit": 0,
                "selection":{
                    "selected": False,
                    "id": 0
                }
            }
            self.visNodes = treeDictionary["visNodes"] if ("visNodes" in treeDictionary) else None
            self.visEdges = treeDictionary["visEdges"] if ("visEdges" in treeDictionary) else []
            self.totalIterations = treeDictionary["totalIterations"] if ("totalIterations" in treeDictionary) else 0
            self.totalRunTime = treeDictionary["totalRunTime"] if ("totalRunTime" in treeDictionary) else 0
            self.graph = treeDictionary["graph"] if ("graph" in treeDictionary) else []
            self.graph2 = treeDictionary["graph2"] if ("graph2" in treeDictionary) else []
            self.height = treeDictionary["height"] if ("height" in treeDictionary) else 0
            self.noIntegerActiveNodes = treeDictionary["noIntegerActiveNodes"] if ("noIntegerActiveNodes" in treeDictionary) else []
            self.error = treeDictionary["error"] if ("error" in treeDictionary) else 0
        else:
            self.maxId = 1
            self.activeNodesIds = None
            self.bestZ = None
            self.bestX = None
            self.bestIter = None
            self.bestTime = None
            self.firstIntegerIter = None
            self.firstIntegerTime = None
            self.type = "min"
            self.typeSearch = "inOrder"
            self.method = "random"
            self.iterator = {
                "type": "unlimited",
                "iteration-steps": 0,
                "time-limit": 0,
                "selection":{
                    "selected": False,
                    "id": 0
                }
            }
            self.visNodes = None
            self.visEdges = []
            self.totalIterations = 0
            self.totalRunTime = 0
            self.graph = []
            self.graph2 = []
            self.height = 0
            self.noIntegerActiveNodes = []
            self.error = 0
        if self.root.mark == "active":
            self.root.solve()
            if self.root.feasible:
                self.activeNodesIds = [{"id": self.root.id,"z":self.root.z}]
                self.visNodes = [{"id": self.root.id,"z":self.root.z,"group":"active","x":self.root.x}]
                if( not self.root.areIntegers()):
                    self.noIntegerActiveNodes = [{"id": self.root.id,"z":self.root.z}]
            else:
                self.activeNodesIds = []
                self.visNodes = [{"id": self.root.id,"z":self.root.z,"group":"infeasible"}]
    
    def getRoot(self):
        return self.root
    
    def serialize(self):
        serialization = {}
        serialization["root"] = self.getRoot().toDictionary()
        serialization["firstIteration"] = self.firstIteration
        serialization["maxId"] = self.maxId
        serialization["activeNodesIds"] = self.activeNodesIds
        serialization["bestZ"] = self.bestZ
        serialization["bestX"] = self.bestX
        serialization["bestIter"] = self.bestIter
        serialization["bestTime"] = self.bestTime
        serialization["firstIntegerIter"] = self.firstIntegerIter
        serialization["firstIntegerTime"] = self.firstIntegerTime
        serialization["type"] = self.type
        serialization["typeSearch"] = self.typeSearch
        serialization["method"] = self.method
        serialization["iterator"] = self.iterator
        serialization["visNodes"] = self.visNodes
        serialization["visEdges"] = self.visEdges
        serialization["totalIterations"] = self.totalIterations
        serialization["totalRunTime"] = self.totalRunTime
        serialization["graph"] = self.graph
        serialization["graph2"] = self.graph2
        serialization["height"] = self.height
        serialization["noIntegerActiveNodes"] = self.noIntegerActiveNodes
        serialization["error"] = self.error
        return serialization

    def searchNode(self, typeSearch="random", typeNode="active"):
        node = None
        if typeSearch=='random':
            node = self.randonMethod()
        elif typeSearch=='inorder':
            node = self.inOrder(self.getRoot(), node, typeNode)
        elif typeSearch=='preorder':
            node = self.preOrder(self.getRoot(), node, typeNode)
        elif typeSearch=='postorder':
            node = self.postOrder(self.getRoot(), node, typeNode)
        elif typeSearch=='bestBounded':
            id = self.activeNodesIds[0]["id"]
            node = self.inOrderSearchId(self.getRoot(), node, id)
        elif typeSearch=='worstBounded':
            id = self.activeNodesIds[len(self.activeNodesIds)-1]["id"]
            node = self.inOrderSearchId(self.getRoot(), node, id)
        elif typeSearch=='selectedNode':
            id = self.iterator["selection"]["id"]
            node = self.inOrderSearchId(self.getRoot(), node, id)
        else:
            node = self.inOrder(self.getRoot(), node, typeNode)
        if node:
            print(f"{typeNode} was found in node {node.id} with method {typeSearch}")
        return node

    def randonMethod(self):
        i = randrange(len(self.activeNodesIds))
        id = self.activeNodesIds[i]["id"]
        node = None
        node = self.inOrderSearchId(self.getRoot(),node,id)
        return node

    def inOrderSearchId(self, node, returnNode, id):
        if node!= None:
            returnNode = self.inOrderSearchId(node.left, returnNode, id)
            if returnNode is not None:
                return returnNode
            if node.id == id:
                return node
            returnNode = self.inOrderSearchId(node.right, returnNode, id)
            if returnNode is not None:
                return returnNode
    
    def inOrder(self, node, returnNode, typeNode="active"):
        if node!= None:
            returnNode = self.inOrder(node.left, returnNode, typeNode)
            if returnNode is not None:
                return returnNode
            if node.mark == typeNode:
                return node
            returnNode = self.inOrder(node.right, returnNode, typeNode)
            if returnNode is not None:
                return returnNode
    
    def preOrder(self, node, returnNode, typeNode="active"):
        if node!= None:
            if node.mark == typeNode:
                return node
            returnNode = self.preOrder(node.left, returnNode, typeNode)
            if returnNode is not None:
                return returnNode
            returnNode = self.preOrder(node.right, returnNode, typeNode)
            if returnNode is not None:
                return returnNode

    def postOrder(self, node, returnNode, typeNode="active"):
        if node!= None:
            returnNode = self.postOrder(node.left, returnNode, typeNode)
            if returnNode is not None:
                return returnNode
            returnNode = self.postOrder(node.right, returnNode, typeNode)
            if returnNode is not None:
                return returnNode
            if node.mark == typeNode:
                return node
    
    def deleteId(self, id):
        filtered = []
        for item in self.activeNodesIds:
            if item["id"] != id:
                filtered.append(item)
        self.activeNodesIds = filtered
    
    def deleteId2(self, id):
        filtered = []
        for item in self.noIntegerActiveNodes:
            if item["id"] != id:
                filtered.append(item)
        self.noIntegerActiveNodes = filtered

    def getZ(self, elem):
        return elem["z"]

    def sortActiveIds(self, method="min-max"):
        if method == "min-max":
            self.activeNodesIds.sort(key=self.getZ)
        if method== "max-min":
            self.activeNodesIds.sort(key=self.getZ,reverse=True)
    
    def sortNoIntegerActiveNodes(self, method="min-max"):
        if method == "min-max":
            self.noIntegerActiveNodes.sort(key=self.getZ)
        if method== "max-min":
            self.noIntegerActiveNodes.sort(key=self.getZ,reverse=True)

    def searchVisNodeAndMarkIt(self, id, mark):
        next(item for item in self.visNodes if item["id"] == id)["group"] = mark

    def updateGraph(self):
        if self.bestZ is not None:
            newData = [
                {
                    "iteration": self.totalIterations,
                    "z": self.bestZ
                }
            ]
            newData = np.array(newData)
            actualData = np.array(self.graph)
            actualData = np.concatenate([actualData, newData])
            self.graph = actualData.tolist()
    
    def updateGraph2(self, iter, score):
        if self.bestZ is not None:
            newData = [
                {
                    "iteration": iter,
                    "z": score
                }
            ]
            newData = np.array(newData)
            actualData = np.array(self.graph2)
            actualData = np.concatenate([actualData, newData])
            self.graph2 = actualData.tolist()

    def branchAndBound(self, typeSearch="random", method="random"):
        # typeSearch = "random"
        # method = "random"
        start_time = time.time()
        self.totalIterations += 1
        print(f"searching for active node with typeSearch {typeSearch}")
        node = self.searchNode(typeSearch, "active")
        if node:
            maxZ = None
            if self.bestZ is None:
                maxZ = float('inf')
            else:
                maxZ = self.bestZ
            if node.z >= maxZ:#self.bestZ:
                if node.areIntegers():
                    node.mark = "optimalBounded"
                    print(f"{node.id} was marked as optimal")
                    self.updateGraph2(self.totalIterations, node.z)
                else:
                    node.mark = "bounded"
                    self.deleteId2(node.id)
                    self.sortNoIntegerActiveNodes("min-max")
                    if(self.bestZ is not None and len(self.noIntegerActiveNodes)>0):
                        self.error = (self.noIntegerActiveNodes[0]["z"]-self.bestZ)/self.bestZ
                        if(self.type=="min"):
                            self.error = (self.bestZ - self.noIntegerActiveNodes[0]["z"])/self.noIntegerActiveNodes[0]["z"]
                    print(f"{node.id} was marked as bounded")
                self.searchVisNodeAndMarkIt(node.id, node.mark)
                self.deleteId(node.id)
                if(len(self.activeNodesIds)==0):
                    self.error = 0
                print(f"Active nodes: {self.activeNodesIds}")
                self.updateGraph()
                self.totalRunTime += time.time()- start_time
                return True
            elif node.areIntegers():
                bestNode = False
                if self.bestZ != None:
                    print(f"searching for best node")
                    bestNode = self.searchNode("inorder","best")
                else:
                    self.firstIntegerIter = self.totalIterations
                    self.firstIntegerTime = self.totalRunTime + time.time()- start_time
                if bestNode:
                    bestNode.mark = "optimal"
                    self.searchVisNodeAndMarkIt(bestNode.id, bestNode.mark)
                    print(f"{bestNode.id} was marked as optimal")
                self.bestZ = node.z
                self.bestX = node.x
                node.mark = "best"
                if(len(self.noIntegerActiveNodes)>0):
                    self.error = (self.noIntegerActiveNodes[0]["z"]-self.bestZ)/self.bestZ
                    if(self.type=="min"):
                            self.error = (self.bestZ - self.noIntegerActiveNodes[0]["z"])/self.noIntegerActiveNodes[0]["z"]
                self.searchVisNodeAndMarkIt(node.id, node.mark)
                print(f"Best was marked at {node.id}")
                self.deleteId(node.id)
                if(len(self.activeNodesIds)==0):
                    self.error = 0
                print(f"Active nodes: {self.activeNodesIds}")
                self.updateGraph()
                self.updateGraph2(self.totalIterations, node.z)
                self.bestIter = self.totalIterations
                self.bestTime = self.totalRunTime + time.time()- start_time
                self.totalRunTime += time.time()- start_time
                return True
            else:
                didBranch = node.branching(self.maxId,method)
                if didBranch:
                    node.mark = "branched"
                    self.searchVisNodeAndMarkIt(node.id, node.mark)
                    print(f"{node.id} was marked as branched with method {method}")
                    newNodesId = []
                    newVisNodes = []
                    newVisEdges = []
                    
                    newVisEdges.append({ "from":node.id , "to": didBranch["leftId"], "varIdx": didBranch["leftVarIdx"], "newValue": didBranch["leftNewValue"], "direction": 'left', "id": len(self.visEdges)+1})
                    if(didBranch["leftFeasible"] == True):
                        newNodesId.append({"id":didBranch["leftId"],"z":didBranch["leftZ"]})
                        newVisNodes.append({"id":didBranch["leftId"],"z":didBranch["leftZ"],"x":didBranch["leftX"], "group":"active", "level":didBranch["height"]})
                    else:
                        newVisNodes.append({"id":didBranch["leftId"],"z":didBranch["leftZ"], "group":"infeasible", "level":didBranch["height"]})
                    
                    newVisEdges.append({ "from":node.id , "to": didBranch["rightId"], "varIdx": didBranch["rightVarIdx"], "newValue": didBranch["rightNewValue"], "direction": 'right', "id": len(self.visEdges)+2})
                    if(didBranch["rightFeasible"] == True):
                        newNodesId.append({"id":didBranch["rightId"],"z":didBranch["rightZ"]})
                        newVisNodes.append({"id":didBranch["rightId"],"z":didBranch["rightZ"],"x":didBranch["rightX"], "group":"active", "level":didBranch["height"]})
                    else:
                        newVisNodes.append({"id":didBranch["rightId"],"z":didBranch["rightZ"], "group":"infeasible", "level":didBranch["height"]})
                    
                    newNodesId = np.array(newNodesId)
                    actualIds = np.array(self.activeNodesIds)
                    actualIds = np.concatenate([actualIds, newNodesId])
                    self.activeNodesIds = actualIds.tolist()

                    newVisNodes = np.array(newVisNodes)
                    actualVisNodes = np.array(self.visNodes)
                    actualVisNodes = np.concatenate([actualVisNodes, newVisNodes])
                    self.visNodes = actualVisNodes.tolist()

                    newVisEdges = np.array(newVisEdges)
                    actualVisEdges = np.array(self.visEdges)
                    actualVisEdges = np.concatenate([actualVisEdges, newVisEdges])
                    self.visEdges = actualVisEdges.tolist()

                    self.deleteId2(node.id)
                    newNoInteger = []
                    if(node.left.feasible and not node.left.areIntegers()):
                        newNoInteger.append({"id":node.left.id,"z":node.left.z})
                        newNoInteger = np.array(newNoInteger)
                        actualNoIntegers = np.array(self.noIntegerActiveNodes)
                        actualNoIntegers = np.concatenate([actualNoIntegers, newNoInteger])
                        self.noIntegerActiveNodes = actualNoIntegers.tolist()
                    newNoInteger = []
                    if(node.right.feasible and not node.right.areIntegers()):
                        newNoInteger.append({"id":node.right.id,"z":node.right.z})
                        newNoInteger = np.array(newNoInteger)
                        actualNoIntegers = np.array(self.noIntegerActiveNodes)
                        actualNoIntegers = np.concatenate([actualNoIntegers, newNoInteger])
                        self.noIntegerActiveNodes = actualNoIntegers.tolist()
                    self.sortNoIntegerActiveNodes("min-max")
                    if(self.bestZ is not None and len(self.noIntegerActiveNodes)>0):
                        self.error = (self.noIntegerActiveNodes[0]["z"]-self.bestZ)/self.bestZ
                        if(self.type=="min"):
                            self.error = (self.bestZ - self.noIntegerActiveNodes[0]["z"])/self.noIntegerActiveNodes[0]["z"]
                        
                    if(didBranch["height"]> self.height):
                        self.height = didBranch["height"]
                    self.maxId = len(self.visNodes)
                    print(f"MaxId: {self.maxId}")
                    self.sortActiveIds("min-max")
                    self.deleteId(node.id)
                    print(f"Active nodes: {self.activeNodesIds}")
                    self.updateGraph()
                    self.totalRunTime += time.time()- start_time
                    return True
                else:
                    node.mark = "error"
                    self.deleteId(node.id)
                    self.updateGraph()
                    self.totalRunTime += time.time()- start_time
                    return False
        else:
            self.updateGraph()
            self.totalRunTime += time.time()- start_time
        return False