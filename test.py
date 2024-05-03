import pandas as pd

# Create a simple dataframe
data = {'A': [1, 2, 3, 4, 5], 'B': [6, 7, 8, 9, 10]}
df = pd.DataFrame(data)


print(df)
# Use isin to find rows where A is in the list [1, 3, 5]
print(df['A'].isin([1, 3, 5]))