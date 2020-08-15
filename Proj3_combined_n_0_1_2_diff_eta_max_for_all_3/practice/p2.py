# importing required modules 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 
import numpy as np 
  
# create a figure, axis and plot element 
fig = plt.figure() 
ax = plt.axes(xlim=(-50, 50), ylim=(-50, 50)) 
line, = ax.plot([], [], lw=2) 
  
# initialization function 
def init(): 
    # creating an empty plot/frame 
    line.set_data([], []) 
    return line, 
  
# lists to store x and y axis points 
xdata, ydata = [], [] 
  
# animation function 
def animate(i): 
    # t is a parameter 
    t = 0.1*i 
      
    # x, y values to be plotted 
    x = t*np.sin(t) 
    y = t*np.cos(t) 
      
    # appending new points to x, y axes points list 
    xdata.append(x) 
    ydata.append(y) 
      
    # set/update the x and y axes data 
    line.set_data(xdata, ydata) 
      
    # return line object 
    return line, 
      
# setting a title for the plot 
plt.title('A growing coil!') 
# hiding the axis details 
plt.axis('off') 
  
# call the animator     
anim = animation.FuncAnimation(fig, animate, init_func=init, 
                               frames=500, interval=20, blit=True) 
  
# save the animation as mp4 video file 
anim.save('animated_coil.mp4', writer = 'ffmpeg', fps = 30) 
  
# show the plot 
plt.show()