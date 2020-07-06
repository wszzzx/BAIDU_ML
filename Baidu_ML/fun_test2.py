import matplotlib.pyplot as plt
list1 = [[1, 2, 3], [4, 5, 6]]
for i in range(len(list1)):
    plt.plot(list1[i],label = i)
plt.legend()
plt.show()