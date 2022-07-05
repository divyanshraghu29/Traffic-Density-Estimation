import numpy as np
import matplotlib.pyplot as plt


def performPlot():
    file1 = open("image_processing/final_count.txt", "r")
    count = file1.readlines()
    # print(count_add)
    x = []
    y = []
    bbox = dict(boxstyle="round", fc="0.8")
    arrowprops = dict(
        arrowstyle="->",
        connectionstyle="angle, angleA = 0, angleB = 90,\
        rad = 10")
    for l in range(0, len(count), 100):
        x.append(l)
        y.append(int(count[l]))
    plt.figure(1, figsize=(7, 5))
    plt.plot(x, y)
    for i in range(0, len(x), 3):
        plt.annotate('(%d, %d)' % (x[i], y[i]),
                     (x[i], y[i]), xytext=(x[i] - 1, y[i] - 1), bbox=bbox, arrowprops=arrowprops)

    # naming the x axis
    plt.xlabel('x - axis-time_frame')
    # naming the y axis
    plt.ylabel('y - axis_number_of_cars')
    # giving a title to my graph
    plt.title('cars_Density')
    plt.tight_layout()
    # function to show the plot
    plt.show()
