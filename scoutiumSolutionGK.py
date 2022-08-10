import pandas as pd
import numpy as np
from termcolor import colored
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected = True)
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,roc_auc_score
pd.set_option('display.width', 600)


sc_attributes = pd.read_csv('../input/scoutium/scoutium_attributes.csv', sep=';')
sc_potential_labels = pd.read_csv('../input/scoutium/scoutium_potential_labels.csv', sep=';')

result = pd.merge(sc_attributes,sc_potential_labels, how="right", on=["task_response_id", "match_id","evaluator_id","player_id"])

df = result.copy()
df.head()

def missing_values_analysis(df):
    na_columns_ = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = df[na_columns_].isnull().sum().sort_values(ascending=False)
    ratio_ = (df[na_columns_].isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio_, 2)], axis=1, keys=['Total Missing Values', 'Ratio'])
    missing_df = pd.DataFrame(missing_df).sort_values(by="Ratio", ascending=False)
    return missing_df


def check_df(df, head=5, tail=5):
    print(" SHAPE ".center(60, '~'))
    print('Observations -------> {}'.format(df.shape[0]))
    print('Features     -------> {}'.format(df.shape[1]))
    print(f"Shape of dataset: {colored(df.shape, 'red')}")
    print(" Types of Features ".center(60, '~'))
    print(df.dtypes,"\n")
    print(" Dataframe - Head ".center(60, '~'))
    print("\n",df.head(head),"\n")
    print(' Dataframe - TAIL '.center(60, '~'))
    print("\n",df.tail(tail),"\n")
    print(" Missing Values Analysis ".center(60, '~'))
    print("\n",missing_values_analysis(df),"\n")
    print(' Duplicate Values Analysis '.center(60, '~'))
    print("\n",df.duplicated().sum(),"\n")
    print(" QUANTILES ".center(60, '~'))
    print("\n",df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T,"\n")


check_df(df)


df.drop(df[df['position_id'] == 1].index, inplace = True)


df[['position_id']].value_counts()

df.drop(df[df['potential_label'] == 'below_average'].index, inplace = True)

df[['potential_label']].value_counts()

print(f"The shape of DataFrame is {colored(df.shape,'red')}")

output = pd.pivot_table(data=df,
                        index=['player_id','position_id','potential_label'],
                        columns=['attribute_id'],
                        values='attribute_value'
                        )
output

output.info()

output.reset_index(inplace=True)
output = output.astype(str)
output

output.info()

def label_encoder(df, column):
    labelencoder = LabelEncoder()
    df[column] = labelencoder.fit_transform(df[column])
    return df

output = label_encoder(output, 'potential_label')
output.head()

output.columns = output.columns.astype(str)
output.columns

num_cols = output.columns[3:]
num_cols

y = output["potential_label"]
X = output.drop(["potential_label", "player_id"], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    random_state = 123,
                                                    stratify = y,
                                                    test_size = 0.2,
                                                    shuffle = True)

print(f"The shape of X_train is --> {colored(X_train.shape,'red')}")
print(f"The shape of X_test is  --> {colored(X_test.shape,'red')}")
print(f"The shape of y_train is --> {colored(y_train.shape,'red')}")
print(f"The shape of y_test is  --> {colored(y_test.shape,'red')}")

def classification_models(model):
    y_pred=model.fit(X_train,y_train).predict(X_test)
    accuracy=accuracy_score(y_pred,y_test)
    roc_score=roc_auc_score(y_pred,model.predict_proba(X_test)[:,1])
    f1=f1_score(y_pred,y_test)
    precision=precision_score(y_pred,y_test)
    recall=recall_score(y_pred,y_test)
    
    results=pd.DataFrame({"Values":[accuracy,roc_score,f1,precision,recall],
                         "Metrics":["Accuracy","ROC-AUC","F1","Precision","Recall"]})
    
    # Visualize Results:
    fig=make_subplots(rows=1,cols=1)
    fig.add_trace(go.Bar(x=[round(i,5) for i in results["Values"]],
                        y=results["Metrics"],
                        text=[round(i,5) for i in results["Values"]],orientation="h",textposition="inside",name="Values",
                        marker=dict(color=["indianred","firebrick","palegreen","skyblue","plum"],line_color="beige",line_width=1.5)),row=1,col=1)
    fig.update_layout(title={'text': model.__class__.__name__ ,
                             'y':0.9,
                             'x':0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'},
                      template='plotly_white')
    fig.update_xaxes(range=[0,1], row = 1, col = 1)

    iplot(fig)

my_models= [
    LogisticRegression(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    GaussianNB()
    ]

for model in my_models:
    classification_models(model)
    

def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

model = GradientBoostingClassifier()
model.fit(X, y)

plot_importance(model, X)
