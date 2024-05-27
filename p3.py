# 3.) For a given set of training data examples stored in a .CSV file, implement and demonstrate the Candidate-Elimination algorithm to output a description of the set of all hypotheses consistent with the training examples.
import numpy as np
import pandas as pd

data = pd.DataFrame(data=pd.read_csv('./p3.csv'))

concepts = np.array(data.iloc[:, 0:-1])  # remove last col from data
target = np.array(data.iloc[:, -1])  # get array of last col

def learn(concepts, target):
    specific_h = concepts[0].copy()
    general_h = [['?' for i in range(len(specific_h))] for i in range(len(specific_h))]
    
    # enumarate (index, value)
    for i, dataRow in enumerate(concepts):
        
        if target[i] == 'Yes':
            for x in range(len(specific_h)):
                if dataRow[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
                    
        if target[i] == 'No':
            for x in range(len(specific_h)):
                if dataRow[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                    
    final_general_h = []
    for row in general_h:
        if row != ['?', '?', '?', '?', '?', '?']:
            final_general_h.append(row)
    return specific_h, final_general_h


s_final, g_final = learn(concepts, target)

print("Final specific_h :", s_final, sep="\n")
print("Final general_h :", g_final, sep="\n")

# OUTPUT :
# Final specific_h :
# ['Sunny' 'Warm' '?' 'Strong' '?' '?']
# Final general_h :
# [['Sunny', '?', '?', '?', '?', '?'], ['?', 'Warm', '?', '?', '?', '?']]
