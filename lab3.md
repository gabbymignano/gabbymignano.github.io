Lab 3 Work:

Question 3:
   ----
      import numpy as np
      import pandas as pd
      from sklearn.model_selection import train_test_split
      from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
      from sklearn.metrics import mean_squared_error, r2_score
   
   ----
      df = pd.read_csv('drive/MyDrive/Data Sci/L3Data.csv')
      y = df['Grade'].values
      cols = ['days online','views','contributions','answers']
      X = df[cols].value
      
   ----
      x_train, x_test, y_train, y_test = train_test_split(X, y, test_size =.25, random_state=1234)
      model = LinearRegression()
      model.fit(x_train,y_train)
      yhat_train= model.predict(x_train)
      yhat_test= model.predict(x_test)
      print('The sq rt MSE for the Test data is: ' +str(np.sqrt(mean_squared_error(y_test, yhat_test))))
            The sq rt MSE for the Test data is: 8.324478857196405

Question 5: Polynomial regression is best suited for functional relationships that are non-linear in weights.
    I wrote True because in Lecture 15 we were unsure about the linearity of the relationship between x and y, yet we still used a polynomial regression.
    However, I now see that it is mentioned in the notes that P(x) is nonlinear in x, but linear in weights.

Question 7:
      ----
      y_train.shape
            (23,)
