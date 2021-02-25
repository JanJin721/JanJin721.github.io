# 결정 트리

- 간단한 결정트리 만들기 실습
- iris 데이터 이용


```python
from sklearn.datasets import load_iris
import pandas as pd

iris_data = load_iris()
x = pd.DataFrame(iris_data.data, columns = iris_data.feature_names)
y = pd.DataFrame(iris_data.target, columns = ['class'])
x.head()
#print(iris_data.DESCR)
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
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 5)
model = DecisionTreeClassifier(max_depth = 4)
model.fit(x_train, y_train)

pred = model.predict(x_test)

model.score(x_test, y_test)
```




    0.9333333333333333




```python
#결정트리에서 모델에서 중요한 변수들을 파악할 수 있음

model.feature_importances_ #변수 순서대로 중요도 출력 
```




    array([0.02551123, 0.        , 0.54152669, 0.43296208])




```python
import matplotlib.pyplot as plt
import numpy as np

importances = model.feature_importances_ #변수 순서대로 중요도 출력 
indices_sorted = np.argsort(importances)

plt.figure()
plt.title('피처 중요도 ')
plt.bar(range(len(importances)), importances[indices_sorted])
plt.xticks(range(len(importances)), x.columns[indices_sorted], rotation =90)
plt.show()

```


    
![png](output_4_0.png)
    


# 랜덤 포레스트
- 랜덤 포레스트 간단하게 만들기
- 마찬가지로 아이리스 데이터 이용


```python
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


iris_data = load_iris()
x = pd.DataFrame(iris_data.data, columns = iris_data.feature_names)
y = pd.DataFrame(iris_data.target, columns = ['class'])

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 5)
y_train = y_train.values.ravel()

model = RandomForestClassifier(n_estimators = 100, max_depth = 4)
model.fit(x_train, y_train)

pred = model.predict(x_test)

model.score(x_test,y_test)
```




    0.9




```python
model.feature_importances_
```




    array([0.0819622 , 0.02053363, 0.40560452, 0.49189965])



# 에다부스트


```python
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier


iris_data = load_iris()
x = pd.DataFrame(iris_data.data, columns = iris_data.feature_names)
y = pd.DataFrame(iris_data.target, columns = ['class'])
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 5)
y_train = y_train.values.ravel()

model = AdaBoostClassifier(n_estimators=1000)
model.fit(x_train,y_train)

score = model.score(x_test,y_test)
score
```




    0.8666666666666667




```python
model.predict(x_test)
```




    array([1, 1, 2, 0, 2, 2, 0, 2, 0, 1, 1, 1, 2, 2, 0, 0, 2, 2, 0, 0, 1, 2,
           0, 1, 1, 2, 1, 1, 1, 2])




```python
model.feature_importances_
```




    array([0.242, 0.003, 0.461, 0.294])


