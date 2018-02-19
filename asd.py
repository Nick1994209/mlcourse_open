import pandas as pd

# df = pd.DataFrame({'A': {0: 'a', 1: 'b', 2: 'c'},
#                    'B': {0: 1, 1: 3, 2: 5},
#                    'C': {0: 2, 1: 4, 2: 6}})
#
# print(pd.melt(df, id_vars='A', value_vars=['B', 'C']))


test = pd.DataFrame({'days': [1, 31, 45, None], 's': [1, 31, None, None]})
# test['range'] = pd.cut(test.days, [0, 30, 60], labels=['a', 'b',])
# print(test)

# print(test)
print(test.loc[:, ['days', 's',]])

print(test[['s', 'days']].dropna())