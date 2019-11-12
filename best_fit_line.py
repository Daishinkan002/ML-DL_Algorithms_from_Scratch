import matplotlib.pyplot as plt
import pandas as pd


#Your Gradient at C function
def get_gradient_at_c(x, y, c, m):
  n = len(x)
  residual = 0
  for i in range(n):
    residual += (y[i] - ((m * x[i]) + c))
  c_gradient = -(2/n) * residual
  return c_gradient

#Your Gradient at M function
def get_gradient_at_m(x, y, c, m):
  n = len(x)
  residual = 0
  for i in range(n):
      residual += x[i] * (y[i] - ((m * x[i]) + c))
  m_gradient = -(2/n) * residual
  return m_gradient


#Your step_gradient function 
def step_gradient(c_current, m_current, x, y,learning_rate):
    c_gradient = get_gradient_at_c(x, y, c_current, m_current)
    m_gradient = get_gradient_at_m(x, y, m_current, m_current)
    c = c_current - (learning_rate * c_gradient)
    m = m_current - (learning_rate * m_gradient)
    return [c, m]


#Your gradient_descent function here:
def gradient_descent(x,y,learning_rate,num_iterations):
  c = 0
  m = 0
  for i in range(num_iterations):
    c,m = step_gradient(c,m,x,y,learning_rate)
  return [c,m]


#For final y after calculating m and c
def calc_y_predicted(x,m,c):
  new_y = [] 
  for i in range(len(x)):
    new_y.append((m*x[i])+c)
  return new_y




n = int(input("Enter total no. of points : "))
x = [0,1,2,3,4,5,6,7,8,9]
y = [1,3,2,5,7,8,8,9,10,12]



for i in range(n):
  x_input = int(input("Enter x " + str(i+1) + " coordinate : "))
  y_input = int(input("Enter  y "+str(i+1)+" coordinate : "))
  x.append(x_input)
  y.append(y_input)
  print()



#learning_rate = float(input("Enter learning_rate : "))
learning_rate = 0.0002
num_iterations = n*500
c,m = gradient_descent(x,y,learning_rate,num_iterations)
new_y = calc_y_predicted(x,m,c)


print("press q to exit")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.plot(x,y,'o')
plt.plot(x,new_y)

plt.plot()
plt.show()
