import numpy as np
import matplotlib.pyplot as plt
base_path = os.path.dirname(os.path.abspath(__file__))
file1_path = os.path.join(base_path, f'count+_.txt')
file2_path = os.path.join(base_path, f'count.txt')

file1=open(file1_path"r")
file2=open("file2_path","r")


#print(count_add)
x=[]
y=[]
def plot():
    count_add=file1.readlines()
    count_sub=file2.readlines()
    for l in range(0,min(len(count_add),len(count_sub)),100):
 
       x.append(l)
       y.append(int(count_add[l])+int(count_sub[l]))

    plt.plot(x, y) 
    
# naming the x axis 
    plt.xlabel('x - axis-time_frame') 
# naming the y axis 
    plt.ylabel('y - axis_number_of_cars') 
    
# giving a title to my graph 
    plt.title('cars_Density') 
    
# function to show the plot 
    plt.show()
 
 if __name__ == '__main__':
     plot()

