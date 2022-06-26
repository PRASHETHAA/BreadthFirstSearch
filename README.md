## EX NO:02
## DATE:28.4.22
# <p align="center">Breadth First Search
## AIM

To develop an algorithm to find the route from the source to the destination point using breadth-first search.

## THEORY
Explain the problem statement

## DESIGN STEPS

### STEP 1:
Identify a location in the google map:

### STEP 2:
Select a specific number of nodes with distance

### STEP -> Write your own steps:


## ROUTE MAP
#### Include your own map
![Output 1](Route.png)

## PROGRAM
```
%matplotlib inline
import matplotlib.pyplot as plt
import random
import math
import sys
from collections import defaultdict, deque, Counter
from itertools import combinations
# Prepared by 
# C. Obed Otto, 
# Department of Artificial Intelligence and Datascience,
# Saveetha Engineering College. 602105. India.
# Experiment done by
# R PRASHETHAA,
# Department of Artificial Intelligence and Data Science,
# Saveetha Engineering College. 602105. India.
#Problems
#This is the abstract class. Specific problem domains will subclass this.
class Problem(object):
    """The abstract class for a formal problem. A new domain subclasses this,
    overriding `actions` and `results`, and perhaps other methods.
    The default heuristic is 0 and the default action cost is 1 for all states.
    When yiou create an instance of a subclass, specify `initial`, and `goal` states 
    (or give an `is_goal` method) and perhaps other keyword args for the subclass."""

    def __init__(self, initial=None, goal=None, **kwds): 
        self.__dict__.update(initial=initial, goal=goal, **kwds) 
        
    def actions(self, state):        
        raise NotImplementedError
    def result(self, state, action): 
        raise NotImplementedError
    def is_goal(self, state):        
        return state == self.goal
    def action_cost(self, s, a, s1): 
        return 1
    
    def __str__(self):
        return '{0}({1}, {2})'.format(
            type(self).__name__, self.initial, self.goal)
#Nodes
#This is the Node in the search tree. Helper functions (expand, path_actions, path_states) use this Node class.
class Node:
    "A Node in a search tree."
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.__dict__.update(state=state, parent=parent, action=action, path_cost=path_cost)

    def __str__(self): 
        return '<{0}>'.format(self.state)
    def __len__(self): 
        return 0 if self.parent is None else (1 + len(self.parent))
    def __lt__(self, other): 
        return self.path_cost < other.path_cost
failure = Node('failure', path_cost=math.inf) # Indicates an algorithm couldn't find a solution.
cutoff  = Node('cutoff',  path_cost=math.inf) # Indicates iterative deepening search was cut off.
#Helper functions
def expand(problem, node):
    "Expand a node, generating the children nodes."
    s = node.state
    for action in problem.actions(s):
        s1 = problem.result(s, action)
        cost = node.path_cost + problem.action_cost(s, action, s1)
        yield Node(s1, node, action, cost)
        

def path_actions(node):
    "The sequence of actions to get to this node."
    if node.parent is None:
        return []  
    return path_actions(node.parent) + [node.action]


def path_states(node):
    "The sequence of states to get to this node."
    if node in (cutoff, failure, None): 
        return []
    return path_states(node.parent) + [node.state]
FIFOQueue = deque
#Search Algorithm : Breadth First Search
def breadth_first_search(problem):
    "Search shallowest nodes in the search tree first."
    node = Node(problem.initial)
    if problem.is_goal(problem.initial):
        return node
    # Remove the following comments to initialize the data structure
    frontier = FIFOQueue([node])
    reached = {problem.initial}
    while frontier:
        node = frontier.pop()
        for child in expand(problem, node):
            s = child.state
            if problem.is_goal(s):
                return child
            if s not in reached:
                reached.add(s)
                frontier.appendleft(child)
    return failure
#Route Finding Problems
class RouteProblem(Problem):
    """A problem to find a route between locations on a `Map`.
    Create a problem with RouteProblem(start, goal, map=Map(...)}).
    States are the vertexes in the Map graph; actions are destination states."""
    
    def actions(self, state): 
        """The places neighboring `state`."""
        return self.map.neighbors[state]
    
    def result(self, state, action):
        """Go to the `action` place, if the map says that is possible."""
        return action if action in self.map.neighbors[state] else state
    
    def action_cost(self, s, action, s1):
        """The distance (cost) to go from s to s1."""
        return self.map.distances[s, s1]
    
    def h(self, node):
        "Straight-line distance between state and the goal."
        locs = self.map.locations
        return straight_line_distance(locs[node.state], locs[self.goal])
class Map:
    """A map of places in a 2D world: a graph with vertexes and links between them. 
    In `Map(links, locations)`, `links` can be either [(v1, v2)...] pairs, 
    or a {(v1, v2): distance...} dict. Optional `locations` can be {v1: (x, y)} 
    If `directed=False` then for every (v1, v2) link, we add a (v2, v1) link."""

    def __init__(self, links, locations=None, directed=False):
        if not hasattr(links, 'items'): # Distances are 1 by default
            links = {link: 1 for link in links}
        if not directed:
            for (v1, v2) in list(links):
                links[v2, v1] = links[v1, v2]
        self.distances = links
        self.neighbors = multimap(links)
        self.locations = locations or defaultdict(lambda: (0, 0))

        
def multimap(pairs) -> dict:
    "Given (key, val) pairs, make a dict of {key: [val,...]}."
    result = defaultdict(list)
    for key, val in pairs:
        result[key].append(val)
    return result
# Create your own map and define the nodes

saveetha_nearby_locations = Map(
    {('Chennai', 'Mathur'):  13, ('Mathur', 'Padianallur'): 10, ('Padianallur', 'Tamaraipakkam'): 17,
     ('Tamaraipakkam', 'Thiruvallur'): 20, ('Thiruvallur', 'Kanakammachathram'): 24, ('Thiruvallur', 'Thiruvalangadu'): 20, 
     ('Thiruvalangadu', 'Thakkolam'):  16, ('Thakkolam', 'Perambakkam'): 12, ('Perambakkam', 'Pannur'): 9, 
     ('Pannur', 'Thakkolam'): 22, ('Perambakkam', 'Polivakkam'): 16, ('Pannur', 'Sriperumbudur'):  13, ('Sriperumbudur', 'Polivakkam'): 12, 
     ('Sriperumbudur', 'Kuthambakkam'): 13, ('Kuthambakkam', 'Poonamallee'): 11, ('Poonamallee','Thandalam'): 13, ('Thandalam','Chromepet'): 10, 
     ('Guindy','Velachery'): 3, ('Guindy','Chennai'): 12, ('Poonamallee','Perumalpattu'): 17, ('Perumalpattu','veppambattu'): 4, 
     ('Perumalpattu','Avadi'): 17, ('Avadi','Ambattur'): 10, 
     ('Ambattur','Chennai'): 16, ('Padianallur','Avadi'): 22})


r0 = RouteProblem('Chennai', 'Padianallur', map=saveetha_nearby_locations)
r1 = RouteProblem('Padianallur', 'Tamaraipakkam', map=saveetha_nearby_locations)
r2 = RouteProblem('Thiruvalangadu', 'Kuthambakkam', map=saveetha_nearby_locations)
r3 = RouteProblem('Kuthambakkam', 'Avadi', map=saveetha_nearby_locations)
r4 = RouteProblem('Avadi', 'Poonamallee', map=saveetha_nearby_locations)
r5 = RouteProblem('Poonamallee', 'Mathur', map=saveetha_nearby_locations)
r6 = RouteProblem('Mathur', 'Kanakammachathram', map=saveetha_nearby_locations)
r7 = RouteProblem('Kanakammachathram', 'Pannur', map=saveetha_nearby_locations)
r8 = RouteProblem('Pannur', 'Chromepet', map=saveetha_nearby_locations)
r9 = RouteProblem('Chromepet', 'veppambattu', map=saveetha_nearby_locations)
r10 = RouteProblem('veppambattu', 'Thakkolam', map=saveetha_nearby_locations)
r11 = RouteProblem('Thakkolam', 'Chennai', map=saveetha_nearby_locations)
r12 = RouteProblem('Pannur', 'Padianallur', map=saveetha_nearby_locations)
r13 = RouteProblem('Tamaraipakkam', 'Ambattur', map=saveetha_nearby_locations)
r14 = RouteProblem('Ambattur', 'Kanakammachathram', map=saveetha_nearby_locations)
r15 = RouteProblem('Velachery', 'Padianallur', map=saveetha_nearby_locations)
print(r0)
print(r1)
print(r2)
print(r3)
print(r4)
print(r5)
print(r6)
print(r7)
print(r8)
print(r9)
print(r10)
print(r11)
print(r12)
print(r13)
print(r14)
print(r15)
goal_state_path=breadth_first_search(r2)
print("GoalStateWithPath:{0}".format(goal_state_path))
path_states(goal_state_path)
print("Total Distance={0} Kilometers".format(goal_state_path.path_cost))
````


## OUTPUT:
![Output2](Output1.png)

![Output3](Output2.png)

![Output4](Output3.png)

![Output5](Output4.png)

## SOLUTION JUSTIFICATION:
Route follow the minimum distance between locations using breadth-first search.
## RESULT:
Thus the program developed for finding route with drawn map and finding its distance covered.
