import numpy as np
import pandas as pd

data = pd.DataFrame(pd.read_csv(r"D:\college\7th sem\AIML lab\7 SEM LAB\7 SEM LAB\p3.csv"))

concepts = np.array(data.iloc[:, 0:-1])
targets = np.array(data.iloc[:, -1])

def learn(concepts, targets):
    specific_h = concepts[0].copy()
    general_h = [['?' for r in range(len(specific_h))] for r in range(len(specific_h))]

    for i, datarow in enumerate(concepts):
        if targets[i] == "Yes":
            for x in range(len(specific_h)):
                if datarow[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'

        if targets[i] == "No":
            for x in range(len(specific_h)):
                if datarow[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]

    final_general_h = []
    for row in general_h:
        if row != ['?', '?', '?', '?', '?', '?']:
            final_general_h.append(row)
    return specific_h, final_general_h


sfinal, s_genaral = learn(concepts, targets)

print("s finalL ", str(sfinal), sep="\n")
print("s general ", str(s_genaral), sep="\n")
