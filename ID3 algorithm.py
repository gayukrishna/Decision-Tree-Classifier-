# Created by: Gayathri Krishnamoorthy
# Updated: 03-23-2020

# ID3 decision tree classifier is implemented here for breast cancer data 
# (https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)classification.
# The information gain heuristic is used for selecting the next feature.
# It is coded in python version 3.6.

## ID3 (https://en.wikipedia.org/wiki/ID3_algorithm) decision tree implementation without pruning 

dataset = pd.read_csv('./breast_cancer/breast-cancer-wisconsin.csv',
                      names=['id_number','clump_thickness','cell_size_uniformity','cell_shape_uniformity','marginal_adhesion',
                             'single_epithelial_cell_size','bare_nuclei','bland_chromatin','normal_nucleoli','mitoses','class'])
dataset=dataset.drop('id_number',axis=1)

###########################################################################################################
##########################################################################################################

def entropy(target_attribute):

    element,count = np.unique(target_attribute,return_counts = True)
    entropy = np.sum([(-count[i]/np.sum(count))*np.log2(count[i]/np.sum(count)) for i in range(len(element))])
    
    return entropy

###########################################################################################################
##########################################################################################################

def infogain(data,selected_attribute,target_attribute="class"):
 
    entropy_first = entropy(data[target_attribute])
    val,count= np.unique(data[selected_attribute],return_counts=True)
    weighted_entropy = np.sum([(count[i]/np.sum(count))*entropy(data.where(data[selected_attribute]==val[i]).dropna()[target_attribute]) for i in range(len(val))])
    information_gain = entropy_first - weighted_entropy
    
    return information_gain
       
###########################################################################################################
##########################################################################################################

def ID3(data,originaldata,features,target_attribute_name="class",parent_node_class = None):

    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    elif len(data)==0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name],return_counts=True)[1])]
     
    elif len(features) ==0:
        return parent_node_class
    
    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],return_counts=True)[1])]
        
        #Select the feature which best splits the dataset
        item_values = [InfoGain(data,feature,target_attribute_name) for feature in features] 
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]        

        tree = {best_feature:{}}        
        
        features = [i for i in features if i != best_feature]
            
        for value in np.unique(data[best_feature]):
            value = value
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = ID3(sub_data,dataset,features,target_attribute_name,parent_node_class)        
            tree[best_feature][value] = subtree
            
        return(tree)    
                
###########################################################################################################
##########################################################################################################   
    
def predict(query,tree,default = 1):
    
    #1.
    for key in list(query.keys()):
        if key in list(tree.keys()):
            #2.
            try:
                result = tree[key][query[key]] 
            except:
                return default
  
            #3.
            result = tree[key][query[key]]
            #4.
            if isinstance(result,dict):
                return predict(query,result)
            else:
                return result
        
###########################################################################################################
##########################################################################################################

def train_valid_test_split(dataset):
    training_data = dataset.iloc[:478].reset_index(drop=True)
    validation_data= dataset.iloc[478:546].reset_index(drop=True)
    testing_data = dataset.iloc[546:].reset_index(drop=True)
    return training_data,validation_data, testing_data

training_data = train_valid_test_split(dataset)[0]
validation_data  = train_valid_test_split(dataset)[1]
testing_data = train_valid_test_split(dataset)[2] 

print(len(training_data))
print(len(validation_data))
print(len(testing_data))

def test(data,tree):
    queries = data.iloc[:,:-1].to_dict(orient = "records")
    predicted = pd.DataFrame(columns=["predicted"]) 
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(queries[i],tree,1.0) 
        
    print('The prediction accuracy is: ',(np.sum(predicted["predicted"] == data["class"])/len(data))*100,'%')
    
###########################################################################################################
##########################################################################################################    
    
## run the ID3 algorithm and test on validation and test data

tree = ID3(training_data,training_data,training_data.columns[:-1])
pprint(tree)
test(validation_data,tree)
test(testing_data,tree)


## ID3 decision tree implementation with pruning 

## testing decision tree using sklearn

#Import the dataset 
dataset = pd.read_csv('./breast_cancer/breast-cancer-wisconsin.csv', names=['id_number','clump_thickness','cell_size_uniformity','cell_shape_uniformity','marginal_adhesion',
                             'single_epithelial_cell_size','bare_nuclei','bland_chromatin','normal_nucleoli','mitoses','class'])

dataset=dataset.drop('id_number',axis=1)
###########################################################################################################
##########################################################################################################

train_features = dataset.iloc[:478,:-1]
validation_features = dataset.iloc[478:546, :-1]
test_features = dataset.iloc[546:, :-1]
train_targets = dataset.iloc[:478, -1]
validation_targets = dataset.iloc[478:546, -1]
test_targets = dataset.iloc[546:, -1]
###########################################################################################################
##########################################################################################################

tree = DecisionTreeClassifier(criterion = 'entropy').fit(train_features,train_targets)
pprint(tree)
###########################################################################################################
##########################################################################################################

prediction_valid = tree.predict(validation_features)
prediction_test = tree.predict(test_features)
###########################################################################################################
##########################################################################################################

print("The validation data prediction accuracy is: ",tree.score(validation_features,validation_targets)*100,"%")
print("The test data prediction accuracy is: ",tree.score(test_features,test_targets)*100,"%")

