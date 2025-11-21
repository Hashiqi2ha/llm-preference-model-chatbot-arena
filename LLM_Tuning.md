```python
import pandas as pd
import numpy as np
import json
```

<div class='alert alert block alert-info'><b>Read data</b></div>


```python
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_df = pd.read_csv('sample_submission.csv')
```


```python
train_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>model_a</th>
      <th>model_b</th>
      <th>prompt</th>
      <th>response_a</th>
      <th>response_b</th>
      <th>winner_model_a</th>
      <th>winner_model_b</th>
      <th>winner_tie</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30192</td>
      <td>gpt-4-1106-preview</td>
      <td>gpt-4-0613</td>
      <td>["Is it morally right to try to have a certain...</td>
      <td>["The question of whether it is morally right ...</td>
      <td>["As an AI, I don't have personal beliefs or o...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>53567</td>
      <td>koala-13b</td>
      <td>gpt-4-0613</td>
      <td>["What is the difference between marriage lice...</td>
      <td>["A marriage license is a legal document that ...</td>
      <td>["A marriage license and a marriage certificat...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>65089</td>
      <td>gpt-3.5-turbo-0613</td>
      <td>mistral-medium</td>
      <td>["explain function calling. how would you call...</td>
      <td>["Function calling is the process of invoking ...</td>
      <td>["Function calling is the process of invoking ...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>96401</td>
      <td>llama-2-13b-chat</td>
      <td>mistral-7b-instruct</td>
      <td>["How can I create a test set for a very rare ...</td>
      <td>["Creating a test set for a very rare category...</td>
      <td>["When building a classifier for a very rare c...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>198779</td>
      <td>koala-13b</td>
      <td>gpt-3.5-turbo-0314</td>
      <td>["What is the best way to travel from Tel-Aviv...</td>
      <td>["The best way to travel from Tel Aviv to Jeru...</td>
      <td>["The best way to travel from Tel-Aviv to Jeru...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>57472</th>
      <td>4294656694</td>
      <td>gpt-4-0613</td>
      <td>claude-1</td>
      <td>["A simple mnemonic for \u03c0:\n\"How I wish ...</td>
      <td>["Sure, let's break it down:\n\n1. \"How\" has...</td>
      <td>["Here is how that mnemonic represents the dig...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>57473</th>
      <td>4294692063</td>
      <td>claude-2.0</td>
      <td>llama-2-13b-chat</td>
      <td>["In python, implement a naive Bayes with gaus...</td>
      <td>["Here is an implementation of a naive Bayes c...</td>
      <td>["Sure! Here's an implementation of a naive Ba...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>57474</th>
      <td>4294710549</td>
      <td>claude-1</td>
      <td>alpaca-13b</td>
      <td>["is it unethical to work on building weapons?...</td>
      <td>["Working on weapons technology raises some et...</td>
      <td>["It depends on the context. Weapons can be us...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>57475</th>
      <td>4294899228</td>
      <td>palm-2</td>
      <td>tulu-2-dpo-70b</td>
      <td>["If a bait contains 0,0025% bromadiolon then ...</td>
      <td>["Bromadiolone is a rodenticide which is most ...</td>
      <td>["As an AI language model, I do not promote or...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>57476</th>
      <td>4294947231</td>
      <td>gemini-pro-dev-api</td>
      <td>gpt-4-1106-preview</td>
      <td>["three kids eat three apples in three days, h...</td>
      <td>["27 apples"]</td>
      <td>["If three kids eat three apples in three days...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>57477 rows × 9 columns</p>
</div>




```python
test_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>prompt</th>
      <th>response_a</th>
      <th>response_b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>136060</td>
      <td>["I have three oranges today, I ate an orange ...</td>
      <td>["You have two oranges today."]</td>
      <td>["You still have three oranges. Eating an oran...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>211333</td>
      <td>["You are a mediator in a heated political deb...</td>
      <td>["Thank you for sharing the details of the sit...</td>
      <td>["Mr Reddy and Ms Blue both have valid points ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1233961</td>
      <td>["How to initialize the classification head wh...</td>
      <td>["When you want to initialize the classificati...</td>
      <td>["To initialize the classification head when p...</td>
    </tr>
  </tbody>
</table>
</div>




```python

sample_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>winner_model_a</th>
      <th>winner_model_b</th>
      <th>winner_tie</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>136060</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>211333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1233961</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_df.isna().sum()
```




    id                0
    model_a           0
    model_b           0
    prompt            0
    response_a        0
    response_b        0
    winner_model_a    0
    winner_model_b    0
    winner_tie        0
    dtype: int64




```python
train_df['winner_model_a'].value_counts()
```




    winner_model_a
    0    37413
    1    20064
    Name: count, dtype: int64



<div class='alert alert block alert-info'><b>Convert Prompt List to String</b></div>


```python
TEXT_COLS = ["prompt", "response_a", "response_b"]

def extract_text(x):
    if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
        try:
            arr = json.loads(x)
            if isinstance(arr, list) and len(arr) > 0:
                return str(arr[0])
        except json.JSONDecodeError:
            pass
    return str(x)

for col in TEXT_COLS:
    train_df[col] = train_df[col].apply(extract_text)
    test_df[col]  = test_df[col].apply(extract_text)
    
train_df[["prompt", "response_a", "response_b"]].head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prompt</th>
      <th>response_a</th>
      <th>response_b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Is it morally right to try to have a certain p...</td>
      <td>The question of whether it is morally right to...</td>
      <td>As an AI, I don't have personal beliefs or opi...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What is the difference between marriage licens...</td>
      <td>A marriage license is a legal document that al...</td>
      <td>A marriage license and a marriage certificate ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>explain function calling. how would you call a...</td>
      <td>Function calling is the process of invoking or...</td>
      <td>Function calling is the process of invoking a ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_df[TEXT_COLS].apply(lambda s: s.str.len().describe())
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prompt</th>
      <th>response_a</th>
      <th>response_b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>57477.000000</td>
      <td>57477.000000</td>
      <td>57477.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>315.817579</td>
      <td>1073.452964</td>
      <td>1079.308245</td>
    </tr>
    <tr>
      <th>std</th>
      <td>937.118852</td>
      <td>879.078526</td>
      <td>888.711302</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>43.000000</td>
      <td>341.000000</td>
      <td>343.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>79.000000</td>
      <td>913.000000</td>
      <td>920.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>194.000000</td>
      <td>1608.000000</td>
      <td>1619.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>12000.000000</td>
      <td>8114.000000</td>
      <td>21443.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
zero_a = (train_df["response_a"].str.len() == 0).sum()
zero_b = (train_df["response_b"].str.len() == 0).sum()
print(f'Response for model A with length of 0: {zero_a}\nResponse for model B with length of 0: {zero_b}')
```

    Response for model A with length of 0: 13
    Response for model B with length of 0: 15
    


```python
label_cols = ["winner_model_a", "winner_model_b", "winner_tie"]
train_df["label_idx"] = np.argmax(train_df[label_cols].values, axis=1)
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>model_a</th>
      <th>model_b</th>
      <th>prompt</th>
      <th>response_a</th>
      <th>response_b</th>
      <th>winner_model_a</th>
      <th>winner_model_b</th>
      <th>winner_tie</th>
      <th>label_idx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30192</td>
      <td>gpt-4-1106-preview</td>
      <td>gpt-4-0613</td>
      <td>Is it morally right to try to have a certain p...</td>
      <td>The question of whether it is morally right to...</td>
      <td>As an AI, I don't have personal beliefs or opi...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>53567</td>
      <td>koala-13b</td>
      <td>gpt-4-0613</td>
      <td>What is the difference between marriage licens...</td>
      <td>A marriage license is a legal document that al...</td>
      <td>A marriage license and a marriage certificate ...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>65089</td>
      <td>gpt-3.5-turbo-0613</td>
      <td>mistral-medium</td>
      <td>explain function calling. how would you call a...</td>
      <td>Function calling is the process of invoking or...</td>
      <td>Function calling is the process of invoking a ...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>96401</td>
      <td>llama-2-13b-chat</td>
      <td>mistral-7b-instruct</td>
      <td>How can I create a test set for a very rare ca...</td>
      <td>Creating a test set for a very rare category c...</td>
      <td>When building a classifier for a very rare cat...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>198779</td>
      <td>koala-13b</td>
      <td>gpt-3.5-turbo-0314</td>
      <td>What is the best way to travel from Tel-Aviv t...</td>
      <td>The best way to travel from Tel Aviv to Jerusa...</td>
      <td>The best way to travel from Tel-Aviv to Jerusa...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
def build_input_row(row):
    return (
        "[PROMPT] " + row["prompt"] +
        " [A] " + row["response_a"] +
        " [B] " + row["response_b"]
    )

train_df["text_all"] = train_df.apply(build_input_row, axis=1)
test_df["text_all"]  = test_df.apply(build_input_row, axis=1)
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>model_a</th>
      <th>model_b</th>
      <th>prompt</th>
      <th>response_a</th>
      <th>response_b</th>
      <th>winner_model_a</th>
      <th>winner_model_b</th>
      <th>winner_tie</th>
      <th>label_idx</th>
      <th>text_all</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30192</td>
      <td>gpt-4-1106-preview</td>
      <td>gpt-4-0613</td>
      <td>Is it morally right to try to have a certain p...</td>
      <td>The question of whether it is morally right to...</td>
      <td>As an AI, I don't have personal beliefs or opi...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[PROMPT] Is it morally right to try to have a ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>53567</td>
      <td>koala-13b</td>
      <td>gpt-4-0613</td>
      <td>What is the difference between marriage licens...</td>
      <td>A marriage license is a legal document that al...</td>
      <td>A marriage license and a marriage certificate ...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>[PROMPT] What is the difference between marria...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>65089</td>
      <td>gpt-3.5-turbo-0613</td>
      <td>mistral-medium</td>
      <td>explain function calling. how would you call a...</td>
      <td>Function calling is the process of invoking or...</td>
      <td>Function calling is the process of invoking a ...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>[PROMPT] explain function calling. how would y...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>96401</td>
      <td>llama-2-13b-chat</td>
      <td>mistral-7b-instruct</td>
      <td>How can I create a test set for a very rare ca...</td>
      <td>Creating a test set for a very rare category c...</td>
      <td>When building a classifier for a very rare cat...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[PROMPT] How can I create a test set for a ver...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>198779</td>
      <td>koala-13b</td>
      <td>gpt-3.5-turbo-0314</td>
      <td>What is the best way to travel from Tel-Aviv t...</td>
      <td>The best way to travel from Tel Aviv to Jerusa...</td>
      <td>The best way to travel from Tel-Aviv to Jerusa...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>[PROMPT] What is the best way to travel from T...</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

X = train_df["text_all"]
y = train_df["label_idx"]

X_tr, X_va, y_tr, y_va = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

tfidf = TfidfVectorizer(
    max_features=200_000,
    ngram_range=(1, 2),
    min_df=2,
)
X_tr_tfidf = tfidf.fit_transform(X_tr)
X_va_tfidf = tfidf.transform(X_va)

clf = LogisticRegression(
    max_iter=200,
    n_jobs=-1,
)
clf.fit(X_tr_tfidf, y_tr)

probs_va = clf.predict_proba(X_va_tfidf)
print("Val log loss:", log_loss(y_va, probs_va))
```

    Val log loss: 1.1063330747559412
    


```python
from sklearn.metrics import classification_report, confusion_matrix

y_va_pred = np.argmax(probs_va, axis=1)

print(classification_report(y_va, y_va_pred, digits=3))
print(confusion_matrix(y_va, y_va_pred))
```

                  precision    recall  f1-score   support
    
               0      0.374     0.408     0.390      4013
               1      0.359     0.368     0.363      3931
               2      0.419     0.363     0.389      3552
    
        accuracy                          0.380     11496
       macro avg      0.384     0.380     0.381     11496
    weighted avg      0.382     0.380     0.381     11496
    
    [[1638 1481  894]
     [1590 1448  893]
     [1156 1108 1288]]
    


```python
pd.set_option("display.max_colwidth", 200) 
va_idx = X_va.index.to_numpy()

conf_scores = probs_va.max(axis=1)
wrong = (y_va_pred != y_va)

bad_idx = np.argsort(conf_scores[wrong])[-5:]
bad_rows = va_idx[wrong][bad_idx]

train_df.loc[bad_rows, ["prompt", "response_a", "response_b", "label_idx"]].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prompt</th>
      <th>response_a</th>
      <th>response_b</th>
      <th>label_idx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>51231</th>
      <td>what's the reverse of string "abcxyz9"</td>
      <td>The reverse of the string "abcxyz9" is "9zyxcba".</td>
      <td>To reverse a string, we can simply swap the characters from the beginning and end of the string. So, the reverse of "abcxyz9" is "9zyxcbax". Therefore, the answer is 9zyxcbax.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>37914</th>
      <td>Today the date is 02.02.2024\nPlease answer all questions in detail, point by point.\n1. Lets talk about an 19 years old european girl, Emese, who is my daughter, with no myopia in her family, who...</td>
      <td>1. It is possible that the environmental factors of COVID restrictions, including reduced outdoor activities and increased screen time, contributed to the onset of Emese's myopia at the age of 16....</td>
      <td>1. It is possible that environmental factors, including increased screen time and reduced outdoor activities during COVID restrictions, could have contributed to the onset or progression of Emese'...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>52043</th>
      <td>Bob (a boy) has 4 sisters. Each sister has 13 brothers. How many brothers does Bob have?</td>
      <td>Bob is a boy, and he has four sisters. Each sister has 13 brothers, but since all siblings share the same set of brothers, this means Bob and his sisters have the same number of brothers. Therefor...</td>
      <td>Since Bob has 4 sisters, and each sister has 13 brothers, we can calculate the total number of brothers by multiplying the number of sisters by the number of brothers each sister has. \n\n4 sister...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>630</th>
      <td>Who is Asia Major</td>
      <td>I'm sorry, but I do not have any information on a person named Asia Major. Can you provide more context or details about who Asia Major is or why you are interested in them?</td>
      <td>I apologize, but I cannot provide information on a person named Asia Major as I couldn't find any relevant or reliable information on them. It's possible that they are a private individual and not...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>55507</th>
      <td>Who is Mohamed sameh gameel</td>
      <td>I'm sorry, but I can't assist with that.</td>
      <td>I'm sorry, but I do not have any specific information on a person named Mohamed Sameh Gameel. There are many people with that name and without more context, I am unable to provide any additional i...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



<div class='alert alert block alert-info'><b>Diaplay rows where model is confident but the prediction is Wrong</b></div>


```python
val_result = pd.DataFrame({
    "idx": X_va.index,
    "label_true": y_va,
    "label_pred": y_va_pred,
    "conf": probs_va.max(axis=1),
})

wrong_mask = val_result["label_true"] != val_result["label_pred"]
wrong_df = val_result[wrong_mask]

top_wrong = wrong_df.sort_values("conf", ascending=False).head(5)

bad_rows = top_wrong["idx"].values

cols_to_show = ["prompt", "response_a", "response_b", "label_idx"]
view_df = train_df.loc[bad_rows, cols_to_show].copy()
view_df["label_pred"] = top_wrong["label_pred"].values

view_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prompt</th>
      <th>response_a</th>
      <th>response_b</th>
      <th>label_idx</th>
      <th>label_pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>55507</th>
      <td>Who is Mohamed sameh gameel</td>
      <td>I'm sorry, but I can't assist with that.</td>
      <td>I'm sorry, but I do not have any specific information on a person named Mohamed Sameh Gameel. There are many people with that name and without more context, I am unable to provide any additional i...</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>630</th>
      <td>Who is Asia Major</td>
      <td>I'm sorry, but I do not have any information on a person named Asia Major. Can you provide more context or details about who Asia Major is or why you are interested in them?</td>
      <td>I apologize, but I cannot provide information on a person named Asia Major as I couldn't find any relevant or reliable information on them. It's possible that they are a private individual and not...</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>52043</th>
      <td>Bob (a boy) has 4 sisters. Each sister has 13 brothers. How many brothers does Bob have?</td>
      <td>Bob is a boy, and he has four sisters. Each sister has 13 brothers, but since all siblings share the same set of brothers, this means Bob and his sisters have the same number of brothers. Therefor...</td>
      <td>Since Bob has 4 sisters, and each sister has 13 brothers, we can calculate the total number of brothers by multiplying the number of sisters by the number of brothers each sister has. \n\n4 sister...</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>37914</th>
      <td>Today the date is 02.02.2024\nPlease answer all questions in detail, point by point.\n1. Lets talk about an 19 years old european girl, Emese, who is my daughter, with no myopia in her family, who...</td>
      <td>1. It is possible that the environmental factors of COVID restrictions, including reduced outdoor activities and increased screen time, contributed to the onset of Emese's myopia at the age of 16....</td>
      <td>1. It is possible that environmental factors, including increased screen time and reduced outdoor activities during COVID restrictions, could have contributed to the onset or progression of Emese'...</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>51231</th>
      <td>what's the reverse of string "abcxyz9"</td>
      <td>The reverse of the string "abcxyz9" is "9zyxcba".</td>
      <td>To reverse a string, we can simply swap the characters from the beginning and end of the string. So, the reverse of "abcxyz9" is "9zyxcbax". Therefore, the answer is 9zyxcbax.</td>
      <td>0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



<div class='alert alert block alert-info'>It appears to me that longer reponse tend to win, creating a table to see if this assumption makes sense</div>


```python
train_df["len_a"] = train_df["response_a"].str.len()
train_df["len_b"] = train_df["response_b"].str.len()
train_df["len_diff"] = train_df["len_a"] - train_df["len_b"]

def get_true_winner(row):
    if row["winner_model_a"] == 1:
        return "A"
    elif row["winner_model_b"] == 1:
        return "B"
    else:
        return "tie"

train_df["winner_true"] = train_df.apply(get_true_winner, axis=1)

def len_bucket(d):
    if d > 100:
        return "A much longer"
    elif d > 0:
        return "A slightly longer"
    elif d < -100:
        return "B much longer"
    elif d < 0:
        return "B slightly longer"
    else:
        return "similar length"

train_df["len_bucket"] = train_df["len_diff"].apply(len_bucket)

true_stats = (
    train_df
    .groupby("len_bucket")["winner_true"]
    .value_counts(normalize=True)
    .rename("ratio")
    .reset_index()
)

true_stats
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>len_bucket</th>
      <th>winner_true</th>
      <th>ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A much longer</td>
      <td>A</td>
      <td>0.457524</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A much longer</td>
      <td>tie</td>
      <td>0.285986</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A much longer</td>
      <td>B</td>
      <td>0.256490</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A slightly longer</td>
      <td>tie</td>
      <td>0.382051</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A slightly longer</td>
      <td>A</td>
      <td>0.316752</td>
    </tr>
    <tr>
      <th>5</th>
      <td>A slightly longer</td>
      <td>B</td>
      <td>0.301197</td>
    </tr>
    <tr>
      <th>6</th>
      <td>B much longer</td>
      <td>B</td>
      <td>0.448203</td>
    </tr>
    <tr>
      <th>7</th>
      <td>B much longer</td>
      <td>tie</td>
      <td>0.287233</td>
    </tr>
    <tr>
      <th>8</th>
      <td>B much longer</td>
      <td>A</td>
      <td>0.264564</td>
    </tr>
    <tr>
      <th>9</th>
      <td>B slightly longer</td>
      <td>tie</td>
      <td>0.380459</td>
    </tr>
    <tr>
      <th>10</th>
      <td>B slightly longer</td>
      <td>B</td>
      <td>0.315553</td>
    </tr>
    <tr>
      <th>11</th>
      <td>B slightly longer</td>
      <td>A</td>
      <td>0.303988</td>
    </tr>
    <tr>
      <th>12</th>
      <td>similar length</td>
      <td>tie</td>
      <td>0.626374</td>
    </tr>
    <tr>
      <th>13</th>
      <td>similar length</td>
      <td>A</td>
      <td>0.192308</td>
    </tr>
    <tr>
      <th>14</th>
      <td>similar length</td>
      <td>B</td>
      <td>0.181319</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>model_a</th>
      <th>model_b</th>
      <th>prompt</th>
      <th>response_a</th>
      <th>response_b</th>
      <th>winner_model_a</th>
      <th>winner_model_b</th>
      <th>winner_tie</th>
      <th>label_idx</th>
      <th>text_all</th>
      <th>len_a</th>
      <th>len_b</th>
      <th>len_diff</th>
      <th>winner_true</th>
      <th>len_bucket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30192</td>
      <td>gpt-4-1106-preview</td>
      <td>gpt-4-0613</td>
      <td>Is it morally right to try to have a certain percentage of females on managerial positions?</td>
      <td>The question of whether it is morally right to aim for a certain percentage of females in managerial positions is a complex ethical issue that involves considerations of fairness, equality, divers...</td>
      <td>As an AI, I don't have personal beliefs or opinions. However, I can tell you that the question of gender quotas in managerial positions is a complex one and positions can vary. Advocates argue tha...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[PROMPT] Is it morally right to try to have a certain percentage of females on managerial positions? [A] The question of whether it is morally right to aim for a certain percentage of females in m...</td>
      <td>2922</td>
      <td>730</td>
      <td>2192</td>
      <td>A</td>
      <td>A much longer</td>
    </tr>
    <tr>
      <th>1</th>
      <td>53567</td>
      <td>koala-13b</td>
      <td>gpt-4-0613</td>
      <td>What is the difference between marriage license and marriage certificate?</td>
      <td>A marriage license is a legal document that allows a couple to get married. It is issued by a government agency, such as a county clerk's office or a state government, and is valid for a certain p...</td>
      <td>A marriage license and a marriage certificate are two different legal documents that have separate purposes.\n\n1. Marriage License: A marriage license is a legal document that gives a couple perm...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>[PROMPT] What is the difference between marriage license and marriage certificate? [A] A marriage license is a legal document that allows a couple to get married. It is issued by a government agen...</td>
      <td>683</td>
      <td>1278</td>
      <td>-595</td>
      <td>B</td>
      <td>B much longer</td>
    </tr>
    <tr>
      <th>2</th>
      <td>65089</td>
      <td>gpt-3.5-turbo-0613</td>
      <td>mistral-medium</td>
      <td>explain function calling. how would you call a function?</td>
      <td>Function calling is the process of invoking or executing a function in a programming language. To call a function, you need to use the function's name followed by parentheses "()". Within the pare...</td>
      <td>Function calling is the process of invoking a function in a program. A function is a block of code that performs a specific task and can be reused multiple times in a program. Functions help to or...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>[PROMPT] explain function calling. how would you call a function? [A] Function calling is the process of invoking or executing a function in a programming language. To call a function, you need to...</td>
      <td>884</td>
      <td>1790</td>
      <td>-906</td>
      <td>tie</td>
      <td>B much longer</td>
    </tr>
    <tr>
      <th>3</th>
      <td>96401</td>
      <td>llama-2-13b-chat</td>
      <td>mistral-7b-instruct</td>
      <td>How can I create a test set for a very rare category? I want to build a classifier.</td>
      <td>Creating a test set for a very rare category can be challenging, as it may not be possible to collect a sufficient number of examples for the rare category. However, there are several strategies y...</td>
      <td>When building a classifier for a very rare category, creating a test set can be challenging due to the limited amount of data available. Here are some steps you can take to create a test set for a...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[PROMPT] How can I create a test set for a very rare category? I want to build a classifier. [A] Creating a test set for a very rare category can be challenging, as it may not be possible to colle...</td>
      <td>3165</td>
      <td>1550</td>
      <td>1615</td>
      <td>A</td>
      <td>A much longer</td>
    </tr>
    <tr>
      <th>4</th>
      <td>198779</td>
      <td>koala-13b</td>
      <td>gpt-3.5-turbo-0314</td>
      <td>What is the best way to travel from Tel-Aviv to Jerusalem? Car? Bus? Plane?</td>
      <td>The best way to travel from Tel Aviv to Jerusalem depends on your personal preference and the availability of transportation options. All the options you have mentioned are valid options, but here...</td>
      <td>The best way to travel from Tel-Aviv to Jerusalem depends on your personal preferences and circumstances. \n\nIf you have a car and enjoy driving, then driving from Tel-Aviv to Jerusalem is an opt...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>[PROMPT] What is the best way to travel from Tel-Aviv to Jerusalem? Car? Bus? Plane? [A] The best way to travel from Tel Aviv to Jerusalem depends on your personal preference and the availability ...</td>
      <td>1290</td>
      <td>760</td>
      <td>530</td>
      <td>B</td>
      <td>A much longer</td>
    </tr>
  </tbody>
</table>
</div>



<div class="alert alert block alert=info"><b>Step 1: Extract Statistical Features</b></div>


```python
def extract_statistical_features(df):
    """
    Extract statistical features from responses
    These features capture quality signals:
    - Length features: longer responses might be more detailed
    - Structure features: lists, code blocks indicate better organization
    - Tone features: apologies, refusals might indicate weak responses
    """
    features = pd.DataFrame(index=df.index)
    
    # === 1. Basic Length Features ===
    print("  - Calculating length features...")
    features['len_a'] = df['response_a'].str.len()
    features['len_b'] = df['response_b'].str.len()
    
    # Length ratio (avoid division by zero)
    features['len_ratio'] = features['len_a'] / (features['len_b'] + 1)
    
    # Length difference
    features['len_diff'] = features['len_a'] - features['len_b']
    
    # Normalized length difference (between -1 and 1)
    features['len_diff_norm'] = features['len_diff'] / (features['len_a'] + features['len_b'] + 1)
    
    # === 2. Word Count Features ===
    print("  - Calculating word count features...")
    features['words_a'] = df['response_a'].str.split().str.len()
    features['words_b'] = df['response_b'].str.split().str.len()
    features['words_ratio'] = features['words_a'] / (features['words_b'] + 1)
    features['words_diff'] = features['words_a'] - features['words_b']
    
    # === 3. Sentence Features ===
    print("  - Calculating sentence features...")
    features['sentences_a'] = df['response_a'].str.count(r'[.!?]+') + 1
    features['sentences_b'] = df['response_b'].str.count(r'[.!?]+') + 1
    
    # Average sentence length (words per sentence)
    features['avg_sent_len_a'] = features['words_a'] / features['sentences_a']
    features['avg_sent_len_b'] = features['words_b'] / features['sentences_b']
    features['avg_sent_diff'] = features['avg_sent_len_a'] - features['avg_sent_len_b']
    
    # === 4. Punctuation Density ===
    print("  - Calculating punctuation features...")
    features['punct_a'] = df['response_a'].str.count(r'[,;:.!?]') / (features['len_a'] + 1)
    features['punct_b'] = df['response_b'].str.count(r'[,;:.!?]') / (features['len_b'] + 1)
    features['punct_diff'] = features['punct_a'] - features['punct_b']
    
    # === 5. Structured Content Features ===
    print("  - Calculating structure features...")
    # Code blocks (marked with ```)
    features['code_blocks_a'] = df['response_a'].str.count('```')
    features['code_blocks_b'] = df['response_b'].str.count('```')
    features['code_diff'] = features['code_blocks_a'] - features['code_blocks_b']
    
    # Bullet points (- * •)
    features['bullets_a'] = df['response_a'].str.count(r'\n[-*•]')
    features['bullets_b'] = df['response_b'].str.count(r'\n[-*•]')
    features['bullets_diff'] = features['bullets_a'] - features['bullets_b']
    
    # Numbered lists (1. 2. 3.)
    features['numbers_a'] = df['response_a'].str.count(r'\n\d+\.')
    features['numbers_b'] = df['response_b'].str.count(r'\n\d+\.')
    features['numbers_diff'] = features['numbers_a'] - features['numbers_b']
    
    # === 6. Tone/Attitude Features ===
    print("  - Calculating tone features...")
    # Apology words (often indicate uncertainty or refusal)
    features['has_sorry_a'] = df['response_a'].str.contains(
        "sorry|apologize|apologies", case=False, regex=True, na=False
    ).astype(int)
    features['has_sorry_b'] = df['response_b'].str.contains(
        "sorry|apologize|apologies", case=False, regex=True, na=False
    ).astype(int)
    features['sorry_diff'] = features['has_sorry_a'] - features['has_sorry_b']
    
    # Refusal words
    features['has_cannot_a'] = df['response_a'].str.contains(
        "cannot|can't|unable|can not", case=False, regex=True, na=False
    ).astype(int)
    features['has_cannot_b'] = df['response_b'].str.contains(
        "cannot|can't|unable|can not", case=False, regex=True, na=False
    ).astype(int)
    features['cannot_diff'] = features['has_cannot_a'] - features['has_cannot_b']
    
    # "I don't know" phrases
    features['has_dontknow_a'] = df['response_a'].str.contains(
        "don't know|do not know|not sure|unclear", case=False, regex=True, na=False
    ).astype(int)
    features['has_dontknow_b'] = df['response_b'].str.contains(
        "don't know|do not know|not sure|unclear", case=False, regex=True, na=False
    ).astype(int)
    features['dontknow_diff'] = features['has_dontknow_a'] - features['has_dontknow_b']
    
    # === 7. Newline count (indicates paragraph structure) ===
    features['newlines_a'] = df['response_a'].str.count('\n')
    features['newlines_b'] = df['response_b'].str.count('\n')
    features['newlines_diff'] = features['newlines_a'] - features['newlines_b']
    
    # Fill any NaN values
    features = features.fillna(0)
    
    print(f"  ✓ Created {features.shape[1]} statistical features")
    return features

# Extract features for train and test sets
train_stat_features = extract_statistical_features(train_df)
test_stat_features = extract_statistical_features(test_df)

# View first few rows
print("\nFeature samples:")
print(train_stat_features.head())

print("\nFeature statistics:")
print(train_stat_features.describe())
```

      - Calculating length features...
      - Calculating word count features...
      - Calculating sentence features...
      - Calculating punctuation features...
      - Calculating structure features...
      - Calculating tone features...
      ✓ Created 38 statistical features
      - Calculating length features...
      - Calculating word count features...
      - Calculating sentence features...
      - Calculating punctuation features...
      - Calculating structure features...
      - Calculating tone features...
      ✓ Created 38 statistical features
    
    Feature samples:
       len_a  len_b  len_ratio  len_diff  len_diff_norm  words_a  words_b  \
    0   2922    730   3.997264      2192       0.600055      415      117   
    1    683   1278   0.534011      -595      -0.303262      121      204   
    2    884   1790   0.493579      -906      -0.338692      147      297   
    3   3165   1550   2.040619      1615       0.342451      547      271   
    4   1290    760   1.695138       530       0.258411      234      124   
    
       words_ratio  words_diff  sentences_a  ...  sorry_diff  has_cannot_a  \
    0     3.516949         298           25  ...           0             0   
    1     0.590244         -83            5  ...           0             0   
    2     0.493289        -150           10  ...           0             0   
    3     2.011029         276           30  ...           0             0   
    4     1.872000         110           11  ...           0             0   
    
       has_cannot_b  cannot_diff  has_dontknow_a  has_dontknow_b  dontknow_diff  \
    0             0            0               0               0              0   
    1             0            0               0               0              0   
    2             0            0               0               0              0   
    3             0            0               0               0              0   
    4             0            0               0               0              0   
    
       newlines_a  newlines_b  newlines_diff  
    0          26           0             26  
    1           0           4             -4  
    2          15          23             -8  
    3          13           8              5  
    4           6           8             -2  
    
    [5 rows x 38 columns]
    
    Feature statistics:
                  len_a         len_b     len_ratio      len_diff  len_diff_norm  \
    count  57477.000000  57477.000000  57477.000000  57477.000000   57477.000000   
    mean    1073.452964   1079.308245      2.551296     -5.855281      -0.001800   
    std      879.078526    888.711302     32.485067    773.798340       0.394152   
    min        0.000000      0.000000      0.000000 -19864.000000      -0.999617   
    25%      341.000000    343.000000      0.616822   -366.000000      -0.233749   
    50%      913.000000    920.000000      0.993007      0.000000       0.000000   
    75%     1608.000000   1619.000000      1.590909    359.000000       0.229323   
    max     8114.000000  21443.000000   4824.000000   6155.000000       0.999793   
    
                words_a       words_b   words_ratio    words_diff   sentences_a  \
    count  57477.000000  57477.000000  57477.000000  57477.000000  57477.000000   
    mean     171.875202    172.599248      1.892051     -0.724046     12.519164   
    std      135.831589    136.349919      6.666291    121.712195     11.098143   
    min        0.000000      0.000000      0.000000  -1511.000000      1.000000   
    25%       57.000000     58.000000      0.607383    -60.000000      4.000000   
    50%      150.000000    151.000000      0.981132      0.000000      9.000000   
    75%      257.000000    259.000000      1.563218     59.000000     18.000000   
    max     1313.000000   1975.000000    673.000000   1058.000000    215.000000   
    
           ...    sorry_diff  has_cannot_a  has_cannot_b   cannot_diff  \
    count  ...  57477.000000  57477.000000  57477.000000  57477.000000   
    mean   ...     -0.000713      0.074517      0.072394      0.002123   
    std    ...      0.245885      0.262612      0.259142      0.328854   
    min    ...     -1.000000      0.000000      0.000000     -1.000000   
    25%    ...      0.000000      0.000000      0.000000      0.000000   
    50%    ...      0.000000      0.000000      0.000000      0.000000   
    75%    ...      0.000000      0.000000      0.000000      0.000000   
    max    ...      1.000000      1.000000      1.000000      1.000000   
    
           has_dontknow_a  has_dontknow_b  dontknow_diff    newlines_a  \
    count    57477.000000    57477.000000   57477.000000  57477.000000   
    mean         0.010700        0.010526       0.000174     11.880300   
    std          0.102886        0.102056       0.138090     15.695016   
    min          0.000000        0.000000      -1.000000      0.000000   
    25%          0.000000        0.000000       0.000000      0.000000   
    50%          0.000000        0.000000       0.000000      8.000000   
    75%          0.000000        0.000000       0.000000     16.000000   
    max          1.000000        1.000000       1.000000    341.000000   
    
             newlines_b  newlines_diff  
    count  57477.000000   57477.000000  
    mean      11.936670      -0.056370  
    std       15.849549      13.503554  
    min        0.000000    -322.000000  
    25%        0.000000      -4.000000  
    50%        8.000000       0.000000  
    75%       16.000000       4.000000  
    max      322.000000     341.000000  
    
    [8 rows x 38 columns]
    

<div class="alert alert block alert=info"><b>Step 2: Combine Statistical Features with TF-IDF</b></div>


```python
from scipy.sparse import hstack
# Get statistical features for train and validation sets
train_stat_va = train_stat_features.loc[X_va.index]
train_stat_tr = train_stat_features.loc[X_tr.index]

# Combine features (TF-IDF is sparse matrix, statistical features are dense)
X_tr_combined = hstack([X_tr_tfidf, train_stat_tr.values])
X_va_combined = hstack([X_va_tfidf, train_stat_va.values])

print(f"Original TF-IDF dimensions: {X_tr_tfidf.shape}")
print(f"Statistical features dimensions: {train_stat_tr.shape}")
print(f"Combined dimensions: {X_tr_combined.shape}")
```

    Original TF-IDF dimensions: (45981, 200000)
    Statistical features dimensions: (45981, 38)
    Combined dimensions: (45981, 200038)
    

<div class="alert alert block alert=info"><b>Step 3: Train Model with Combined Features</b></div>


```python
clf_improved = LogisticRegression(
    max_iter=500,
    C=0.5,  # Slightly more regularization
    n_jobs=-1,
    random_state=42
)

print("Training...")
clf_improved.fit(X_tr_combined, y_tr)

# Predict
probs_va_improved = clf_improved.predict_proba(X_va_combined)
y_va_pred_improved = np.argmax(probs_va_improved, axis=1)

# Evaluate
from sklearn.metrics import log_loss, classification_report, accuracy_score

logloss_improved = log_loss(y_va, probs_va_improved)
accuracy_improved = accuracy_score(y_va, y_va_pred_improved)

print(f"\n{'='*60}")
print("Model Comparison:")
print(f"{'='*60}")
print(f"Original Model Log Loss: 1.1063")
print(f"Improved Model Log Loss: {logloss_improved:.4f}")
print(f"\nOriginal Model Accuracy: 38.0%")
print(f"Improved Model Accuracy: {accuracy_improved*100:.1f}%")
print(f"{'='*60}")

print("\nDetailed Classification Report:")
print(classification_report(y_va, y_va_pred_improved, 
                          target_names=['Model A wins', 'Model B wins', 'Tie'],
                          digits=3))
```

    Training...
    
    ============================================================
    Model Comparison:
    ============================================================
    Original Model Log Loss: 1.1063
    Improved Model Log Loss: 1.0623
    
    Original Model Accuracy: 38.0%
    Improved Model Accuracy: 44.6%
    ============================================================
    
    Detailed Classification Report:
                  precision    recall  f1-score   support
    
    Model A wins      0.441     0.612     0.513      4013
    Model B wins      0.449     0.571     0.503      3931
             Tie      0.462     0.122     0.193      3552
    
        accuracy                          0.446     11496
       macro avg      0.451     0.435     0.403     11496
    weighted avg      0.450     0.446     0.411     11496
    
    

<div class="alert alert block alert=info"><b>Step 4: Analyze Feature Importance</b></div>


```python
# Get coefficients for statistical features (skip TF-IDF features)
stat_feature_names = train_stat_features.columns.tolist()
n_tfidf_features = X_tr_tfidf.shape[1]

# Get coefficients for statistical features (average absolute value across 3 classes)
stat_coef = clf_improved.coef_[:, n_tfidf_features:]  # Only statistical features
feature_importance = np.abs(stat_coef).mean(axis=0)

# Create feature importance DataFrame
importance_df = pd.DataFrame({
    'feature': stat_feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Statistical Features:")
print(importance_df.head(15).to_string(index=False))
```

    
    Top 15 Most Important Statistical Features:
          feature  importance
       sorry_diff    0.010503
      words_ratio    0.009477
      cannot_diff    0.007178
      has_sorry_b    0.006875
      sentences_a    0.006793
      has_sorry_a    0.006562
      sentences_b    0.005883
    code_blocks_b    0.005405
        code_diff    0.004859
        bullets_a    0.004709
     has_cannot_b    0.004664
     has_cannot_a    0.004660
    code_blocks_a    0.004165
    len_diff_norm    0.003675
     numbers_diff    0.003620
    

<div class="alert alert block alert=info"><b>Step 5: Analyze Improved Error Cases</b></div>


```python
val_result_improved = pd.DataFrame({
    "idx": X_va.index,
    "label_true": y_va,
    "label_pred": y_va_pred_improved,
    "conf": probs_va_improved.max(axis=1),
})

wrong_mask_improved = val_result_improved["label_true"] != val_result_improved["label_pred"]
wrong_df_improved = val_result_improved[wrong_mask_improved]

print(f"Original model errors: {wrong.sum()}")
print(f"Improved model errors: {wrong_mask_improved.sum()}")
print(f"Reduction in errors: {wrong.sum() - wrong_mask_improved.sum()} samples")

# View high-confidence errors
top_wrong_improved = wrong_df_improved.sort_values("conf", ascending=False).head(5)
bad_rows_improved = top_wrong_improved["idx"].values

cols_to_show = ["prompt", "response_a", "response_b", "label_idx"]
view_df_improved = train_df.loc[bad_rows_improved, cols_to_show].copy()
view_df_improved["label_pred"] = top_wrong_improved["label_pred"].values

print("\nHigh-confidence errors in improved model:")
print(view_df_improved)
```

    Original model errors: 7122
    Improved model errors: 6364
    Reduction in errors: 758 samples
    
    High-confidence errors in improved model:
                                                                                                                                                                                                            prompt  \
    37310  Aerodynamics, a branch of fluid dynamics, is the study of the motion of air, particularly when it interacts with a solid object, such as an airplane wing. It involves the analysis of forces and th...   
    21011  Consider this list of ANZSIC codes for companies: Nursery Production (Under Cover)\nNursery Production (Outdoors)\nTurf Growing\nFloriculture Production (Under Cover)\nFloriculture Production (Out...   
    21701  A concise introduction to logic fourteenth edition by Patrick Hurley, 1.1 exercise set 1 problems 15-30. Each of the following passages contains a single argument. Using the letters “P” and\n“C,” ...   
    54008  Re-write the story from Nagumo's POV:\n\nPrologue\nLight rapidly faded away as the darkness began to swallow him.\nHajime Nagumo’s face twisted in terror as he stared at the vanishing\nlight above...   
    34403  How could I build a VST plugin uses AI to mix, master and EQ my tracks based on common production styles I like (e.g. Daft Punk, Beatles, Mac Demarco)? Ideally I'd use a pre-trained transformer mo...   
    
                                                                                                                                                                                                        response_a  \
    37310  What does Aerofield create?\n\nAerofield creates an area of non-aeroplasticity, given fluid induced vortices. This means that in the presence of fluid-induced vortices, the Aerofield establishes a...   
    21011  This is a list of ANZSIC (Australian and New Zealand Standard Industrial Classification) codes for various industries and activities. Here's a summary of the sectors represented in the list:\n\n1....   
    21701  I'm sorry, but I cannot provide the solutions to the exercises as I do not have access to the specific textbook or the problems mentioned. However, I can help you understand the process of identif...   
    54008  ## Prologue\n\nThe darkness enveloped Hajime Nagumo as he fell through the chasm, the light above him disappearing. His life flashed before his eyes, accompanied by the sound of wind rushing past....   
    34403  To build a VST plugin that uses AI to mix, master, and EQ your tracks based on common production styles, you can follow these steps:\n\n1. Segment the audio into its different tracks (vocals, guit...   
    
                                                                                                                                                                                                        response_b  \
    37310  Aerodynamics is the study of the motion of air, particularly when it interacts with a solid object, such as an airplane wing. It involves the analysis of forces and the resulting motion caused by ...   
    21011  It looks like your message was cut off. Could you provide more details or clarify how I can assist you with the list of ANZSIC codes you've posted? If you have specific questions about these codes...   
    21701  Sure! Here are the arguments and their identified premises and conclusions for exercise set 1 problems 15-30:\n\n15. "The cost of a college education is rising faster than the rate of inflation. T...   
    54008                                                                                                                       I'm sorry, but I'm not sure what you're asking me to do. Could you please clarify?   
    34403  This is an extremely broad and ambitious goal. First, you need to collect a large dataset of audio samples with different production features. Then you need to build a model to transform spectrogr...   
    
           label_idx  label_pred  
    37310          0           1  
    21011          1           0  
    21701          2           1  
    54008          2           0  
    34403          2           0  
    

<div class="alert alert block alert=info"><b>Step 6: Train LightGBM Model (More Powerful than Logistic Regression)</b></div>


```python
import lightgbm as lgb
from sklearn.metrics import log_loss, accuracy_score, classification_report

# Keep sparse format - DON'T convert to dense!
print(f"Training set shape: {X_tr_combined.shape}")
print(f"Validation set shape: {X_va_combined.shape}")

# LightGBM parameters
lgb_params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

# Create datasets
print("\nCreating LightGBM datasets...")
train_data = lgb.Dataset(X_tr_combined, label=y_tr)
valid_data = lgb.Dataset(X_va_combined, label=y_va, reference=train_data)

# Train model with proper callbacks
print("\nTraining LightGBM...")
callbacks = [
    lgb.early_stopping(stopping_rounds=50),
    lgb.log_evaluation(period=100)
]

lgb_model = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, valid_data],
    valid_names=['train', 'valid'],
    callbacks=callbacks
)

print(f"\nBest iteration: {lgb_model.best_iteration}")

# Predict
probs_va_lgb = lgb_model.predict(X_va_combined, num_iteration=lgb_model.best_iteration)
y_va_pred_lgb = np.argmax(probs_va_lgb, axis=1)

# Evaluate
logloss_lgb = log_loss(y_va, probs_va_lgb)
accuracy_lgb = accuracy_score(y_va, y_va_pred_lgb)

print(f"\n{'='*60}")
print("Model Performance Comparison:")
print(f"{'='*60}")
print(f"Original LogReg Log Loss:  1.1063 | Accuracy: 38.0%")
print(f"Improved LogReg Log Loss:  1.0623 | Accuracy: 44.0%")
print(f"LightGBM        Log Loss:  {logloss_lgb:.4f} | Accuracy: {accuracy_lgb*100:.1f}%")
print(f"{'='*60}")

print("\nDetailed Classification Report (LightGBM):")
print(classification_report(y_va, y_va_pred_lgb,
                          target_names=['Model A wins', 'Model B wins', 'Tie'],
                          digits=3))

# Error analysis
wrong_mask_lgb = (y_va_pred_lgb != y_va)
print(f"\nError Comparison:")
print(f"Original LogReg errors: 7,122")
print(f"Improved LogReg errors: 6,364")
print(f"LightGBM errors:        {wrong_mask_lgb.sum():,}")
print(f"Reduction from improved LogReg: {6364 - wrong_mask_lgb.sum():,} samples")
```

    Training set shape: (45981, 200038)
    Validation set shape: (11496, 200038)
    
    Creating LightGBM datasets...
    
    Training LightGBM...
    Training until validation scores don't improve for 50 rounds
    [100]	train's multi_logloss: 0.94983	valid's multi_logloss: 1.03477
    Early stopping, best iteration is:
    [103]	train's multi_logloss: 0.947438	valid's multi_logloss: 1.03468
    
    Best iteration: 103
    

    C:\Users\65748\anaconda3\envs\llm-gpu\Lib\site-packages\lightgbm\basic.py:1238: UserWarning: Converting data to scipy sparse matrix.
      _log_warning("Converting data to scipy sparse matrix.")
    

    
    ============================================================
    Model Performance Comparison:
    ============================================================
    Original LogReg Log Loss:  1.1063 | Accuracy: 38.0%
    Improved LogReg Log Loss:  1.0623 | Accuracy: 44.0%
    LightGBM        Log Loss:  1.0347 | Accuracy: 47.1%
    ============================================================
    
    Detailed Classification Report (LightGBM):
                  precision    recall  f1-score   support
    
    Model A wins      0.478     0.540     0.507      4013
    Model B wins      0.483     0.517     0.499      3931
             Tie      0.440     0.342     0.385      3552
    
        accuracy                          0.471     11496
       macro avg      0.467     0.466     0.464     11496
    weighted avg      0.468     0.471     0.467     11496
    
    
    Error Comparison:
    Original LogReg errors: 7,122
    Improved LogReg errors: 6,364
    LightGBM errors:        6,084
    Reduction from improved LogReg: 280 samples
    

<div class="alert alert block alert=info">Feature Importance for LightGBM</div>


```python
# Get feature importance
importance = lgb_model.feature_importance(importance_type='gain')

# Get feature names
tfidf_feature_names = [f'tfidf_{i}' for i in range(X_tr_tfidf.shape[1])]
all_feature_names = tfidf_feature_names + train_stat_features.columns.tolist()

# Create importance dataframe
importance_lgb_df = pd.DataFrame({
    'feature': all_feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

# Show top features
print("\nTop 20 Most Important Features:")
print(importance_lgb_df.head(20).to_string(index=False))

# Show top statistical features only
stat_importance = importance_lgb_df[
    importance_lgb_df['feature'].isin(train_stat_features.columns)
]
print("\nTop 15 Most Important Statistical Features:")
print(stat_importance.head(15).to_string(index=False))
```

    
    Top 20 Most Important Features:
          feature   importance
         len_diff 18116.697983
    newlines_diff  5152.791809
       words_diff  4108.177486
      tfidf_10072  2258.016176
      has_sorry_b  1493.463522
      has_sorry_a  1353.640718
    len_diff_norm  1298.763311
       sorry_diff   951.545544
       punct_diff   851.855551
     tfidf_174159   615.014778
        len_ratio   611.291276
     tfidf_177456   458.323852
       newlines_b   418.281981
       newlines_a   379.557783
     tfidf_135290   374.392479
      tfidf_77504   368.927500
     tfidf_134497   358.433728
    avg_sent_diff   354.340870
      tfidf_52490   319.027598
      cannot_diff   318.602800
    
    Top 15 Most Important Statistical Features:
          feature   importance
         len_diff 18116.697983
    newlines_diff  5152.791809
       words_diff  4108.177486
      has_sorry_b  1493.463522
      has_sorry_a  1353.640718
    len_diff_norm  1298.763311
       sorry_diff   951.545544
       punct_diff   851.855551
        len_ratio   611.291276
       newlines_b   418.281981
       newlines_a   379.557783
    avg_sent_diff   354.340870
      cannot_diff   318.602800
        code_diff   317.626851
     bullets_diff   299.396188
    

<div class="alert alert block alert=info">Analyze Remaining Difficult Cases</div>


```python
val_result_lgb = pd.DataFrame({
    "idx": X_va.index,
    "label_true": y_va,
    "label_pred": y_va_pred_lgb,
    "conf": probs_va_lgb.max(axis=1),
})

wrong_df_lgb = val_result_lgb[val_result_lgb["label_true"] != val_result_lgb["label_pred"]]

# High confidence errors
top_wrong_lgb = wrong_df_lgb.sort_values("conf", ascending=False).head(5)
bad_rows_lgb = top_wrong_lgb["idx"].values

view_df_lgb = train_df.loc[bad_rows_lgb, ["prompt", "response_a", "response_b", "label_idx"]].copy()
view_df_lgb["label_pred"] = top_wrong_lgb["label_pred"].values
view_df_lgb["confidence"] = top_wrong_lgb["conf"].values

print("\nTop 5 High-Confidence Errors:")
print(view_df_lgb[["prompt", "label_idx", "label_pred", "confidence"]])
```

    
    Top 5 High-Confidence Errors:
                                                                                                                                                                                                            prompt  \
    6940   My grandma forgot the recipe to her favorite cocktail. how to make a it? It had the following stuff:\n\nSugar\nWater\nCoke\nAspartame\ngramp's "sweet and sour cadmium chunks  made with love"\nAlso...   
    37363                                                                                   how much were black slaves for and was their selling reasonable? What's the average iq of a black race representitive.   
    32896                                                                                                                                      There are 9 eggs and 6 cups of water.\n\nSuggest a way to lay eggs.   
    46930                                                                                                                                  my mom is very dominant. do you think she would be a good domme for me?   
    41272                                                                                                                                                        List the special story modules in shadow empire\n   
    
           label_idx  label_pred  confidence  
    6940           0           1    0.847444  
    37363          2           1    0.818055  
    32896          2           1    0.806041  
    46930          2           1    0.800877  
    41272          0           1    0.795305  
    

<div class="alert alert block alert=info">Plot Learning Curves</div>

<div class="alert alert block alert=info"><b>Step 7: Add Sentence-BERT Embeddings</b></div>


```python
from sentence_transformers import SentenceTransformer
import torch

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load pre-trained model (this is one of the best models for semantic similarity)
print("\nLoading Sentence-BERT model...")
sbert_model = SentenceTransformer('all-mpnet-base-v2', device=device)
print("✓ Model loaded successfully")
```

    Using device: cuda
    
    Loading Sentence-BERT model...
    ✓ Model loaded successfully
    

<div class="alert alert block alert=info">Create Embeddings for All Text</div>


```python
def create_embeddings(df, model, batch_size=32):
    """
    Create embeddings for prompt, response_a, and response_b
    Then create comparison features
    """
    print(f"  Processing {len(df)} samples...")
    
    # Encode all texts
    prompts = df['prompt'].tolist()
    responses_a = df['response_a'].tolist()
    responses_b = df['response_b'].tolist()
    
    print("  - Encoding prompts...")
    prompt_emb = model.encode(prompts, batch_size=batch_size, 
                             show_progress_bar=True, convert_to_numpy=True)
    
    print("  - Encoding response A...")
    response_a_emb = model.encode(responses_a, batch_size=batch_size,
                                  show_progress_bar=True, convert_to_numpy=True)
    
    print("  - Encoding response B...")
    response_b_emb = model.encode(responses_b, batch_size=batch_size,
                                  show_progress_bar=True, convert_to_numpy=True)
    
    # Create comparison features
    print("  - Creating comparison features...")
    
    # Cosine similarity between prompt and responses
    from sklearn.metrics.pairwise import cosine_similarity
    
    sim_prompt_a = np.array([cosine_similarity([p], [r])[0][0] 
                            for p, r in zip(prompt_emb, response_a_emb)])
    sim_prompt_b = np.array([cosine_similarity([p], [r])[0][0] 
                            for p, r in zip(prompt_emb, response_b_emb)])
    
    # Similarity difference
    sim_diff = sim_prompt_a - sim_prompt_b
    
    # Euclidean distance
    dist_a = np.linalg.norm(prompt_emb - response_a_emb, axis=1)
    dist_b = np.linalg.norm(prompt_emb - response_b_emb, axis=1)
    dist_diff = dist_a - dist_b
    
    # Direct embedding differences (response_a - response_b)
    response_diff = response_a_emb - response_b_emb
    
    # Concatenate all features
    embedding_features = np.column_stack([
        prompt_emb,           # 768 dims
        response_a_emb,       # 768 dims
        response_b_emb,       # 768 dims
        response_diff,        # 768 dims
        sim_prompt_a.reshape(-1, 1),
        sim_prompt_b.reshape(-1, 1),
        sim_diff.reshape(-1, 1),
        dist_a.reshape(-1, 1),
        dist_b.reshape(-1, 1),
        dist_diff.reshape(-1, 1),
    ])
    
    print(f"  ✓ Created {embedding_features.shape[1]} embedding features")
    return embedding_features

# Generate embeddings for train and test
print("\nGenerating embeddings for training set...")
train_embeddings = create_embeddings(train_df, sbert_model, batch_size=32)

print("\nGenerating embeddings for test set...")
test_embeddings = create_embeddings(test_df, sbert_model, batch_size=32)
```

    
    Generating embeddings for training set...
      Processing 57477 samples...
      - Encoding prompts...
    

    Batches: 100%|██████████| 1797/1797 [01:16<00:00, 23.57it/s]
    

      - Encoding response A...
    

    Batches: 100%|██████████| 1797/1797 [04:27<00:00,  6.73it/s]
    

      - Encoding response B...
    

    Batches: 100%|██████████| 1797/1797 [04:30<00:00,  6.65it/s]
    

      - Creating comparison features...
      ✓ Created 3078 embedding features
    
    Generating embeddings for test set...
      Processing 3 samples...
      - Encoding prompts...
    

    Batches: 100%|██████████| 1/1 [00:00<00:00, 17.47it/s]
    

      - Encoding response A...
    

    Batches: 100%|██████████| 1/1 [00:00<00:00,  3.60it/s]
    

      - Encoding response B...
    

    Batches: 100%|██████████| 1/1 [00:00<00:00, 14.07it/s]

      - Creating comparison features...
      ✓ Created 3078 embedding features
    

    
    

<div class="alert alert block alert=info">Split embeddings into train/validation</div>


```python
# Get embeddings for train and validation splits
train_emb_tr = train_embeddings[X_tr.index]
train_emb_va = train_embeddings[X_va.index]

# Combine with statistical features
X_tr_full = np.column_stack([
    train_emb_tr,
    train_stat_tr.values
])

X_va_full = np.column_stack([
    train_emb_va,
    train_stat_va.values
])

print(f"Embedding features: {train_emb_tr.shape[1]}")
print(f"Statistical features: {train_stat_tr.shape[1]}")
print(f"Total features: {X_tr_full.shape[1]}")
print(f"Training samples: {X_tr_full.shape[0]}")
print(f"Validation samples: {X_va_full.shape[0]}")
```

    Embedding features: 3078
    Statistical features: 38
    Total features: 3116
    Training samples: 45981
    Validation samples: 11496
    

<div class="alert alert block alert=info">Train LightGBM with Full Feature Set</div>


```python
# Optimized parameters for larger feature set
lgb_params_v2 = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 64,  # Increased for more complex features
    'learning_rate': 0.03,  # Slightly lower for stability
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'reg_alpha': 0.1,  # L1 regularization
    'reg_lambda': 0.1,  # L2 regularization
    'verbose': -1,
    'random_state': 42
}

# Create datasets
train_data_v2 = lgb.Dataset(X_tr_full, label=y_tr)
valid_data_v2 = lgb.Dataset(X_va_full, label=y_va, reference=train_data_v2)

# Train with early stopping
print("Training...")
evals_result_v2 = {}
lgb_model_v2 = lgb.train(
    lgb_params_v2,
    train_data_v2,
    num_boost_round=2000,
    valid_sets=[train_data_v2, valid_data_v2],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=200),
        lgb.record_evaluation(evals_result_v2)
    ]
)

print(f"Best iteration: {lgb_model_v2.best_iteration}")
```

    Training...
    [200]	train's multi_logloss: 0.766162	valid's multi_logloss: 1.02394
    [400]	train's multi_logloss: 0.579208	valid's multi_logloss: 1.02258
    Best iteration: 416
    

<div class="alert alert block alert=info">Evaluate Final Model</div>


```python
# Predict
probs_va_final = lgb_model_v2.predict(X_va_full, num_iteration=lgb_model_v2.best_iteration)
y_va_pred_final = np.argmax(probs_va_final, axis=1)

# Calculate metrics
logloss_final = log_loss(y_va, probs_va_final)
accuracy_final = accuracy_score(y_va, y_va_pred_final)

print(f"\n{'='*70}")
print("COMPLETE MODEL PERFORMANCE COMPARISON")
print(f"{'='*70}")
print(f"{'Model':<35} {'Log Loss':<12} {'Accuracy':<12}")
print(f"{'-'*70}")
print(f"{'Original LogReg (TF-IDF only)':<35} {'1.1063':<12} {'38.0%':<12}")
print(f"{'LogReg + Statistical Features':<35} {'1.0623':<12} {'44.0%':<12}")
print(f"{'LightGBM + Statistical Features':<35} {'1.0347':<12} {'47.1%':<12}")
print(f"{'LightGBM + SBERT + Statistical':<35} {f'{logloss_final:.4f}':<12} {f'{accuracy_final*100:.1f}%':<12}")
print(f"{'='*70}")

improvement = accuracy_final - 0.471
print(f"\n🚀 Improvement from LightGBM: +{improvement*100:.1f} percentage points")
print(f"🚀 Total improvement from baseline: +{(accuracy_final - 0.38)*100:.1f} percentage points")
```

    
    ======================================================================
    COMPLETE MODEL PERFORMANCE COMPARISON
    ======================================================================
    Model                               Log Loss     Accuracy    
    ----------------------------------------------------------------------
    Original LogReg (TF-IDF only)       1.1063       38.0%       
    LogReg + Statistical Features       1.0623       44.0%       
    LightGBM + Statistical Features     1.0347       47.1%       
    LightGBM + SBERT + Statistical      1.0222       47.9%       
    ======================================================================
    
    🚀 Improvement from LightGBM: +0.8 percentage points
    🚀 Total improvement from baseline: +9.9 percentage points
    


```python
print("\nDetailed Classification Report:")
print(classification_report(y_va, y_va_pred_final,
                          target_names=['Model A wins', 'Model B wins', 'Tie'],
                          digits=3))

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_va, y_va_pred_final)
print("\nConfusion Matrix:")
print("                Predicted:")
print("                A wins  B wins  Tie")
print(f"Actual: A wins  {cm[0,0]:<7} {cm[0,1]:<7} {cm[0,2]:<7}")
print(f"        B wins  {cm[1,0]:<7} {cm[1,1]:<7} {cm[1,2]:<7}")
print(f"        Tie     {cm[2,0]:<7} {cm[2,1]:<7} {cm[2,2]:<7}")
```

    
    Detailed Classification Report:
                  precision    recall  f1-score   support
    
    Model A wins      0.482     0.536     0.507      4013
    Model B wins      0.489     0.533     0.510      3931
             Tie      0.460     0.355     0.401      3552
    
        accuracy                          0.479     11496
       macro avg      0.477     0.475     0.473     11496
    weighted avg      0.477     0.479     0.475     11496
    
    
    Confusion Matrix:
                    Predicted:
                    A wins  B wins  Tie
    Actual: A wins  2150    1095    768    
            B wins  1121    2096    714    
            Tie     1193    1098    1261   
    

<div class="alert alert block alert=info"><b>Generate Predictions on Test Set</b></div>


```python
test_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>prompt</th>
      <th>response_a</th>
      <th>response_b</th>
      <th>text_all</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>136060</td>
      <td>I have three oranges today, I ate an orange yesterday. How many oranges do I have?</td>
      <td>You have two oranges today.</td>
      <td>You still have three oranges. Eating an orange yesterday does not affect the number of oranges you have today.</td>
      <td>[PROMPT] I have three oranges today, I ate an orange yesterday. How many oranges do I have? [A] You have two oranges today. [B] You still have three oranges. Eating an orange yesterday does not af...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>211333</td>
      <td>You are a mediator in a heated political debate between two opposing parties. Mr Reddy is very hung up on semantic definitions of sex and gender, and believes that women are adult human females. M...</td>
      <td>Thank you for sharing the details of the situation. As a mediator, I understand the importance of finding a middle ground that both parties can agree upon.\n\nIt is important to recognize that lan...</td>
      <td>Mr Reddy and Ms Blue both have valid points in their arguments. On one hand, Mr Reddy is correct that women are adult human females, as defined by their biological and physical characteristics. Ho...</td>
      <td>[PROMPT] You are a mediator in a heated political debate between two opposing parties. Mr Reddy is very hung up on semantic definitions of sex and gender, and believes that women are adult human f...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1233961</td>
      <td>How to initialize the classification head when I do transfer learning. For example, I have a pre-trained vision transformer on ImageNet, and now I want to finetune it on StanfordCars</td>
      <td>When you want to initialize the classification head for transfer learning, you can follow these steps:\n\n  1. Load the pre-trained model: First, load the pre-trained vision transformer model on I...</td>
      <td>To initialize the classification head when performing transfer learning, follow these steps:\n\n1. Load the pre-trained vision transformer model that was trained on ImageNet.\n2. Replace the exist...</td>
      <td>[PROMPT] How to initialize the classification head when I do transfer learning. For example, I have a pre-trained vision transformer on ImageNet, and now I want to finetune it on StanfordCars [A] ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_test_tfidf = tfidf.transform(test_df["text_all"])
X_test_combined = hstack([X_test_tfidf, test_stat_features.values])

# Predict
test_probs = lgb_model.predict(X_test_combined, num_iteration=lgb_model.best_iteration)

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'winner_model_a': test_probs[:, 0],
    'winner_model_b': test_probs[:, 1],
    'winner_tie': test_probs[:, 2]
})

pd.DataFrame(submission)
```

    C:\Users\65748\anaconda3\envs\llm-gpu\Lib\site-packages\lightgbm\basic.py:1238: UserWarning: Converting data to scipy sparse matrix.
      _log_warning("Converting data to scipy sparse matrix.")
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>winner_model_a</th>
      <th>winner_model_b</th>
      <th>winner_tie</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>136060</td>
      <td>0.266982</td>
      <td>0.327763</td>
      <td>0.405255</td>
    </tr>
    <tr>
      <th>1</th>
      <td>211333</td>
      <td>0.542357</td>
      <td>0.186305</td>
      <td>0.271338</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1233961</td>
      <td>0.281101</td>
      <td>0.471340</td>
      <td>0.247559</td>
    </tr>
  </tbody>
</table>
</div>


