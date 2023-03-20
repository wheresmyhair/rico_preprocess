# rico_preprocess
``` python
for example in dataset: 
    node = example.rootNode  
    while not example.isTerminalNode(node): 
    optimal = set() 
    if len(node.children)==0: 
        # at a leaf node, go back up 
        optimal.add(PopAction()) 
    else: 
        for child in node.children: 
            if child.isLeaf: 
                optimal.add(ArcAction(child)) 
            else: 
                optimal.add(EmitAction()) 
    if dynamicTraining: 
        model.train(optimal) 
        if model.highestScoring in optimal: 
            action = model.prediction 
        else: 
            action=randomChoice(optimal) 
    else: # static training 
        action=getCanonicalAction(optimal) 
        model.train(action) 
    node=node.nodeAfterAction(action)
```