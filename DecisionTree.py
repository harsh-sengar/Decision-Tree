from sklearn import datasets as d
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import math


def label(val, *boundaries): #functions to convert continous data to labelled data
    if (val < boundaries[0]):
        return 'a'
    elif (val < boundaries[1]):
        return 'b'
    elif (val < boundaries[2]):
        return 'c'
    else:
        return 'd'



def toLabel(df, old_feature_name):
    second = df[old_feature_name].mean()
    minimum = df[old_feature_name].min()
    first = (minimum + second)/2
    maximum = df[old_feature_name].max()
    third = (maximum + second)/2
    return df[old_feature_name].apply(label, args= (first, second, third))



class node:
    def __init__(self, n, ent, pent, l, df, y):
        self.name=n
        self.bestFeature = None
        self.entropy = ent
        self.parentEntropy = pent
        self.level=l
        self.children = []
        self.dataframe = df
        self.target = y
        
    def addchild(self,obj):
        self.children.append(obj)
        
    def getentropy(self):
        return self.entropy
    
    def getlevel(self):
        return self.level
        


def countTrue(df,feature_name,val):
    x = df[feature_name] == val
    count = 0
    for i in x:
        if i == True:
            count+=1
    return count


# In[34]:


def countY(y, val):
    count = 0
    for i in y:
        if i == val:
            count+=1
    return count



def entropyCalc(y):
    pc = y.apply(pd.value_counts)
    pc = np.array(pc)
    total = pc.sum()
    #print(total)
    entropy = 0
    for i in pc:
        #print (i)
        if(i>0):
            entropy = entropy + (-1*(i/total)*math.log((i/total),2))
        #print(entropy)
    return entropy   



def weightedEntropy(df, y,feature):
    if(feature == None):
        return entropyCalc(y)
    
    values = set(df[feature])
    
    val = []  #putting distinct values in a list because set can be accessed by index
    for i in values:
        val.append(i)
    #print(val)
    #caluculating weights of distinct values in df[feature] and putting them in dictionary weight
        
    weight = {}
    for i in val:
        weight[i] = countTrue(df,feature,i)/df[feature].shape[0]
    #print(weight)
    # putting entropy corresponding to distinct values in a certain feature in dictionary called entropy
    
    entropy = {}    
    for i in val:
        entropy[i] = entropyCalc(y[df[feature]==i])
    #print(entropy)
    
    #calculating weighted entropy... 
        
    weighted_entropy = 0
    
    for i in val:
        weighted_entropy = weighted_entropy + (weight[i]*entropy[i])
        
    return weighted_entropy
    
    

def informationGain(parent_entropy, child_entropy):
    return parent_entropy - child_entropy



def distinctVal(y):
    distinct = set(y)
    return len(distinct)
    


def buildTree(df, y, unused_features, cur_node):
    #base case
    # 1. unused is empty
    # 2. y contains only one distinct value
    if(len(unused_features)==0 or distinctVal(y.output) == 1):
         return

    #at the end of for loop best_feature will contain the name of feature along which on splitting there will be max info gain.
    best_feature = ""
    max_info_gain = 0
    
    for f in unused_features:        
        weighted_entropy = weightedEntropy(df, y, f)
        #print(weighted_entropy)
        #since cur_node will act as parent oon splitting along selected feature
        parent_entropy =cur_node.getentropy()      
        info_gain = informationGain(parent_entropy, weighted_entropy)
        if(info_gain > max_info_gain):
            max_info_gain = info_gain
            best_feature = f
            
    print("Best Feature ", best_feature)  
    unused_features.remove(best_feature)
    possible_values = set(df[best_feature])
    cur_node.bestFeature= best_feature
    for val in possible_values:
        #print(val)
        new_y=y[df[best_feature]==val]
        new_df=df[df[best_feature]==val]
        new_ent=entropyCalc(new_y)  
        #creating a node corresponding to each of the distinct values of df[best_feature]
        new_node=node(val,new_ent,cur_node.getentropy(),cur_node.getlevel()+1,new_df,new_y)
        new_node.bestFeature = best_feature
        cur_node.addchild(new_node)
        
    for child in cur_node.children:
        new_df = child.dataframe
        new_y = child.target
        buildTree(new_df,new_y,unused_features,child)
    
    # remove best feature from unused features
    # loop over possible values of best feature
    # call build tree recursively



def decisionTreeBuilder(df, y, features):
    #since root has no parent, keeping its parent's entropy same as that of the root
    current_node = node('root', entropyCalc(y), entropyCalc(y), 0, df, y)
    
    #call build tree
    buildTree(df, y, features, current_node)
    
    print("Level ",current_node.level)
    possible_outputs = set(current_node.target.output)
    for x in possible_outputs:
        print("Count of ",x," = ",countY(current_node.target.output, x))
    print("Current Entropy is = ",current_node.getentropy())
    weighted_entropy = weightedEntropy(current_node.dataframe, current_node.target, current_node.bestFeature)
    print("Splitting on feature ",current_node.bestFeature," with gain ratio ",informationGain(current_node.parentEntropy, weighted_entropy))
    print()
    
    printTree(current_node)



def printTree(current_node):#dfs
    for child in current_node.children:
        print("Level ",child.level)
        ss=list(child.target)
        f=False
        if(child.bestFeature != None):
                f=True
            
  
        possible_outputs = set(child.target[ss[0]])
        lst=list(child.target[ss[0]])
        for val in possible_outputs:
            print("Count of ",val," = ",countY(lst,val))
        print("Current Entropy  is = ",child.getentropy())
        
        if(f==True and child.getentropy()!=0):
            weighted_entropy = weightedEntropy(child.dataframe,child.target,child.bestFeature)
            info_gain = informationGain(current_node.entropy, weighted_entropy)
            print("Splitting on feature ",child.bestFeature," with gain ratio ",info_gain)
            print()
        elif(f==False or child.getentropy()==0):
            print("Reached leaf Node")
            print()
        printTree(child)
         
     

def DecisionTreesonIris():
    iris = d.load_iris()
    df = pd.DataFrame(iris.data)
    y = pd.DataFrame(iris.target)

    #giving headers to data columns
    df.columns = ["sl", "sw", "pl", "pw"]
    y.columns = ['output']
    
    #converting continuous data to labelled data
    df['sl_labeled'] = toLabel(df, 'sl')
    df['sw_labeled'] = toLabel(df, 'sw')
    df['pl_labeled'] = toLabel(df, 'pl')
    df['pw_labeled'] = toLabel(df, 'pw')
    
    #dropping original columns
    df.drop(['sl', 'sw', 'pl', 'pw'], axis = 1, inplace = True)
    
    #feature list
    attributes = []
    for j in df.columns:
        attributes.append(j)
    #splitting data
    #x_train, x_test, y_train, y_test = train_test_split(df, y)
    
    #call decision tree builder
    decisionTreeBuilder(df, y, attributes)
   # print(y)



DecisionTreesonIris()

