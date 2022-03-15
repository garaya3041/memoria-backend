import json, falcon, numpy, time
from scipy.optimize import linprog
from tree import Node, Tree

class ObjRequestClass:
    def on_post(self, req, resp):
        ans = Tree(req.media)
        start_time = time.time()
        actual_time = time.time()
        count = 0
        if (not ans.firstIteration):
            if (ans.iterator["type"] == "unlimited"):
                while len(ans.activeNodesIds)>0:
                    ans.branchAndBound(ans.typeSearch, ans.method)
            elif (ans.iterator["type"] == "timer"):
                while actual_time - start_time < ans.iterator["time-limit"] and len(ans.activeNodesIds)>0:
                    ans.branchAndBound(ans.typeSearch, ans.method)
                    actual_time = time.time()
            elif (ans.iterator["type"] == "steps"):
                while count < ans.iterator["iteration-steps"] and len(ans.activeNodesIds)>0:
                    ans.branchAndBound(ans.typeSearch, ans.method)
                    count += 1
            else:
                while len(ans.activeNodesIds)>0:
                    ans.branchAndBound(ans.typeSearch, ans.method)
        jsonTree = ans.serialize()
        resp.body = json.dumps(jsonTree)

api = falcon.API(cors_enable=True)
api.add_route('/optimize', ObjRequestClass())