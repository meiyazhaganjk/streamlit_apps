import streamlit as st
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.rc('font',family='serif',size='5')

def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)
  
    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)
  
    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
  
    # calculating regression coefficients
    w_auto = SS_xy / SS_xx
    b_auto = m_y - w_auto*m_x
  
    return (w_auto, b_auto)


st.title("Linear Regression")
st.sidebar.write("$y = w x + b $")


n = st.number_input('Enter the number of data', min_value=5)

col1, col2 = st.columns(2)

xs, ys = [], []
with col1:
   st.header("Enter the value of x")
   for i in range(0,int(n)):
    gg = st.number_input('x',key='x_input'+str(i), step=0.5, label_visibility="hidden")
    xs.append(gg)

with col2:
   st.header("Enter the value of y")
   for i in range(0,int(n)):
    ff = st.number_input('y',key='y_input'+str(i),  step=0.5, label_visibility="hidden")
    ys.append(ff)

x = np.array(xs)
y = np.array(ys)

if st.button('Show Data Table'):
    df = pd.DataFrame(np.concatenate((x,y)).reshape(x.shape[0],2),
        columns=("x", "y"))
    st.table(df)

fig1 = plt.figure(figsize =(3, 3)) 
ax = fig1.add_subplot(111)
ax.scatter(x, y, color='green', s=10)
ax.set_xlabel('x')
ax.set_ylabel('y')

w = st.sidebar.slider("Value of Weigth 'w'", 0.0, 5.0, 0.1)
b = st.sidebar.slider("Value of Bias 'b'", 0.0, 5.0, 0.1)
y_pre_manual = w*x + b

ax.plot(x, y_pre_manual, 'b-', label="Manual")




auto_button = st.sidebar.button('Start Automatic fixing')

if auto_button:
    w_auto, b_auto = estimate_coef(x, y)
    st.sidebar.write("The value of $w$ is ", w_auto)
    st.sidebar.write("The value of $b$ is ", b_auto)
    y_auto = w_auto*x + b_auto
    ax.plot(x, y_auto, "r-", label="Auto fitting")

ax.set_xlim(min(x)-0.5, max(x)+0.5)
ax.set_ylim(min(x)-0.5, max(y)+0.5)
ax.set_xticks([i for i in range(int(min(x)), int(max(x))+1, 3)])
ax.set_yticks([i for i in range(int(min(y)), int(max(y))+1, 3)])
ax.legend(loc='best')
st.pyplot(fig1)
