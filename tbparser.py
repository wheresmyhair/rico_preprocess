import json


path_file = './data/json/0.json'
with open(path_file, 'r') as f:
    data = json.load(f)


# In transition-based parsers, the oracle function is used during training 
# to determine the correct sequence of transitions that should be taken to 
# produce the correct parse tree for a given sentence. 


# The oracle function takes as input a partially constructed parse tree, 
# represented as a stack and a buffer, and the correct parse tree for the 
# current sentence. It then returns the next transition that should be taken 
# in order to most closely resemble the correct parse tree. 


# During training, the parser is run on a corpus of sentences and the oracle 
# function is used to provide the correct transitions, allowing the parser to 
# learn how to construct parse trees from raw text.


# In this example, the `stack` represents the partially constructed parse tree 
# in the form of a stack, and the `buffer` represents the remaining words that 
# still need to be parsed. The `deps` dictionary contains the current dependency 
# relations in the parse tree, and `gold_deps` contains the correct relations for 
# the current sentence.

# The function first checks if there are two items on the stack, and if so, 
# it checks if a valid left-arc or right-arc operation can be performed based 
# on the gold dependencies. If not, it checks if there are items remaining in 
# the buffer and performs a shift operation to move the first item to the top 
# of the stack.


# If the stack has only one item and there are no items in the buffer, or if 
# no valid operations can be performed, the function returns `None`, indicating 
# that the parser is stuck and cannot proceed.


# In this example, the `train()` function takes in a `parser` object (which should 
# have an `update()` method), the number of `epochs` to train for, and a `data` 
# list containing tuples of sentence, tags, and gold dependencies.


# For each sentence in the training data, the function initializes an empty `stack`, 
# `buffer`, and `deps` dictionary. It then iterates until the stack has only one item 
# and the buffer is empty, using the `get_oracle()` function to determine the next 
# transition based on the current state and the gold dependencies.


# If a valid transition is found, the function applies it to the stack, buffer, and 
# dependencies. After processing the entire sentence, the function updates the parser 
# weights based on the predicted dependencies and the gold dependencies using the 
# `parser.update()` method.


# By repeating this process for multiple epochs and multiple sentences in the 
# training data, the parser learns to predict the correct sequence of transitions 
# for constructing parse trees from raw text.


def get_oracle(stack, buffer, deps, gold_deps):
    # If the stack has at least 2 items
    if len(stack) > 1:
        # Check if the topmost item of the stack has a head
        top = stack[-1]
        second = stack[-2]
        if top in gold_deps and gold_deps[top] == second:
            # If there is a gold dependency between the topmost item and the second topmost item,
            # perform a LEFT-ARC operation to make the topmost item the dependent of the second topmost item
            return 'larc'
        
        if second in gold_deps and gold_deps[second] == top:
            # If there is a gold dependency between the second topmost item and the topmost item,
            # perform a RIGHT-ARC operation to make the topmost item the head of the second topmost item
            return 'rarc'
          
        if len(buffer) > 0:
            # If there are items remaining in the buffer,
            # perform a SHIFT operation to move the first item in the buffer to the top of the stack
            return 'shift'
        
    else:
        if len(buffer) > 0:
            # If the stack has only one item and there are items in the buffer,
            # perform a SHIFT operation to move the first item in the buffer to the top of the stack
            return 'shift'
          
    # No valid operations, the parser is stuck
    return None

def train(parser, epochs, data):
    for i in range(epochs):
        for sentence, tags, gold_deps in data:
            # Initialize empty stack, buffer, and dependencies
            stack = []
            buffer = list(enumerate(sentence))
            deps = {}

            # Iterate until the stack has only one item and the buffer is empty
            while len(buffer) > 0 or len(stack) > 1:
                # Get the current oracle transition based on the stack, buffer, and gold dependencies
                transition = get_oracle(stack, buffer, deps, gold_deps)

                if transition is None:
                    # If there is no valid transition, bail out of the loop
                    break

                # Apply the transition to the stack, buffer, and dependencies
                if transition == 'shift':
                    stack.append(buffer.pop(0))
                elif transition == 'larc':
                    deps[stack[-1]] = stack[-2]
                    stack.pop(-2)
                elif transition == 'rarc':
                    deps[stack[-2]] = stack[-1]
                    stack.pop(-1)

            # Update the parser weights based on the predicted dependencies and gold dependencies
            parser.update(deps, gold_deps)



# for example in dataset: 
#     node = example.rootNode  
#     while not example.isTerminalNode(node): 
#         optimal = set() 
#         if len(node.children)==0: 
#             # at a leaf node, go back up 
#             optimal.add(PopAction()) 
#         else: 
#             for child in node.children: 
#                 if child.isLeaf: 
#                     optimal.add(ArcAction(child)) 
#                 else: 
#                     optimal.add(EmitAction()) 
#         if dynamicTraining: 
#             model.train(optimal) 
#             if model.highestScoring in optimal: 
#                 action = model.prediction 
#             else: 
#                 action=randomChoice(optimal) 
#         else: # static training 
#             action=getCanonicalAction(optimal) 
#             model.train(action) 
#         node=node.nodeAfterAction(action)

class UIHierarchy(object): 
    def __init__(self, hierarchy):
        self.rootNode = hierarchy['activity']['root']
        
class Node(object):
    def __init__(self, node):
        self.node = node
        self.children = node['children']
        
    def is_terminal(self):
        return len(self.children) == 0
    
    def after_action(self, action):
        # TBD
        if action == 'pop':
            return self.node['parent']
        elif action == 'arc':
            return self.children[0]
        else:
            return self.node

class Optimal(list):
    def __init__(self):
        super(Optimal, self).__init__()
        
    def add_arc_action(self, arc):
        self.append(arc)
        
    def add_emit_action(self):
        self.append('emit')
        
    def add_pop_action(self):
        self.append('pop')
        
    def get_canonical_action(self):
        if 'pop' in self:
            return 'pop'
        elif 'arc' in self:
            return 'arc'
        else:
            return 'emit'
    
example = UIHierarchy(data)
node = Node(example.rootNode)

while not node.is_terminal():
    # init
    optimal = Optimal()
    
    # get optimal actions
    if node.is_terminal():
        optimal.add_pop_action()
    else:
        for child in node.children:
            subnode = Node(child)
            if subnode.is_terminal():
                optimal.add_arc_action(child) # change to id
            else:
                optimal.add_emit_action()
                
    # TODO: Training
    # if dynamicTraining: 
    #     model.train(optimal) 
    #     if model.highestScoring in optimal: 
    #         action = model.prediction 
    #     else: 
    #         action=randomChoice(optimal) 
    # else: # static training 
    #     action = optimal.get_canonical_action()
    #     model.train(action) 
    action = optimal.get_canonical_action()
        
    # update node
    node = node.after_action(action)