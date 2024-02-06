import mlxtend
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

transactions = [['Milk', 'Bread', 'Butter'],
                ['Milk', 'Bread'],
                ['Milk', 'Diapers'],
                ['Milk', 'Beer', 'Diapers'],
                ['Bread', 'Butter'],
                ['Bread', 'Beer'],
                ['Butter', 'Diapers'],
                ['Bread', 'Beer', 'Diapers']]

encoder = TransactionEncoder()
onehot = encoder.fit_transform(transactions)
df = pd.DataFrame(onehot, columns=encoder.columns_)

frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

print("Frequent Itemsets:")
print(frequent_itemsets)

print("Association Rules:")
print(rules)
