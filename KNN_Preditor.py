import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


data = torch.load(r"C:\Users\JR100\OneDrive\Airfoil\Airfoil\data\naca4_simple_multiple_airfoils\data.pt")


# data[0:6] recieves the airfoil type 2412
x = np.concatenate([i.x for i in data[0:6]], axis=0 )
y = np.concatenate([ i.y for i in data[0:6]],axis =0)
pressure = np.concatenate([ i.pressure for i in data[0:6]], axis=0)



# y will act as our input 
x_train,x_test, y_train,y_test = train_test_split(y, pressure, test_size= .30, shuffle= True)




# neighbors of 11 was arbitrarily choosen but is odd to eliminate ties
nn_r = KNeighborsRegressor(n_neighbors=11, weights="distance",  n_jobs=-1)
pred= nn_r.fit(x_train,y_train).predict(x_test)


plt.scatter(x_test,y_test, linewidths= 3, label= "Actual Pressure")
plt.scatter (x_test,pred, color= "red", linewidths= .01, label="Predicted Pressure")

plt.legend(["Actual Pressure","Predicted Pressure"])
plt.xlabel("Y Component")
plt.ylabel ("Pressure")
plt.show()











