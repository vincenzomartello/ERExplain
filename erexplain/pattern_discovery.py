import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth,fpmax,association_rules



def _createTransactions(df,columns,class_to_explain,left_prefix,right_prefix):
    transactions = []
    for i in range(len(df)):
        leftValues,rightValues = [],[]
        for attr in columns:
            if attr.startswith(right_prefix):
                leftValues += str(df.iloc[i][attr]).split()
            elif attr.startswith(right_prefix):
                rightValues += str(df.iloc[i][attr]).split()
        if class_to_explain == 0:
            selectedRightValues = set(leftValues).intersection(set(rightValues))
            selectedLeftValues = selectedRightValues.copy()
        else:
            selectedLeftValues = set(leftValues).difference(set(rightValues))
            selectedRightValues = set(rightValues).difference(set(leftValues))
        leftValuesPrefixed = list(map(lambda val:'L_'+val,selectedLeftValues))
        rightValuesPrefixed = list(map(lambda val:'R_'+val,selectedRightValues))
        transactions.append(leftValuesPrefixed+rightValuesPrefixed)
    return transactions


def mineAssociationRules(df,columns,class_to_explain,lprefix='ltable_',rprefix='rtable_',min_confidence=0.5,min_support=0.2):
	transactions = _createTransactions(df,columns,class_to_explain,left_prefix=lprefix,right_prefix=rprefix)
	te = TransactionEncoder()
	te_ary = te.fit(transactions).transform(transactions)
	df = pd.DataFrame(te_ary, columns=te.columns_)
	frequent_itemsets = fpmax(df, min_support=min_support,use_colnames=True)
	ar = association_rules(frequent_itemsets, metric="confidence", min_threshold = min_confidence)
	ar['antecedents_isleft'] = ar['antecedents'].apply(lambda s: all(token.startswith('L_') for token in s))
	ar['consequents_isright'] = ar['consequents'].apply(lambda s: all(token.startswith('R_') for token in s))
	important_rules = ar[(ar.antecedents_isleft==True)& (ar.consequents_isright==True)]
	return important_rules


def getMaxFrequentPatterns(df,columns,class_to_explain,lprefix='ltable_',rprefix='rtable_',min_support=0.2,k=15):
    transactions = _createTransactions(df,columns,class_to_explain,left_prefix=lprefix,right_prefix=rprefix)
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = fpmax(df, min_support=min_support,use_colnames=True)
    return frequent_itemsets
