This script can be used for stratifying multi-label data. It is based on work by: Sechidis, Konstantinos and Tsoumakas, Grigorios and Vlahavas, Ioannis: On the stratification of multi-label data. In: Proceedings of the 2011 European conference on Machine learning and knowledge discovery in databases (ECML PKDD'11), volume part III, pp. 145-158 (2011)
                                     
Here is a small example on how to use:

```python
import numpy as np

import stratified_multilabel_KFold


if __name__ == "__main__":
    
    X = np.array([[1, 2], 
                  [3, 4], 
                  [1, 2], 
                  [3, 4],
                  [2, 3], 
                  [1, 4], 
                  [3, 4],  
                  [2, 4], 
                  [1, 5]])
    
    y = np.array([[1,0,1], 
                  [0,0,1], 
                  [0,1,0], 
                  [1,0,0],
                  [0,1,1], 
                  [1,1,0], 
                  [1,0,1],  
                  [1,0,1], 
                  [0,0,1]])
    
    
    cv = stratified_multilabel_KFold.stratified_multilabel_KFold(n_splits = 3, shuffle = False)
    
    for train_index, test_index in cv.split(y):
        print("TRAIN:", train_index, "TEST:", test_index)
        
        X_train, X_test = X[train_index,:], X[test_index,:]
        y_train, y_test = y[train_index,:], y[test_index,:]
```
