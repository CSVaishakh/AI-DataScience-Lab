import pandas as pd

path = "sample_data.csv"

df = pd.read_csv(path)

data = df.values.tolist()

sortLeader = 'Age'

leaderIndex = list(df.columns).index(sortLeader)

n = len(df)

for i in range(n):
    for j in range(0,n-i-1):
        row1 = data[j]
        row2 = data[j+1]
        if row1[leaderIndex] > row2[leaderIndex]:
            data[j],data[j+1] = row2, row1

sorted_df = pd.DataFrame(data,columns=df.columns)

sorted_df.to_csv('sorted_sample_data.csv', index=False)