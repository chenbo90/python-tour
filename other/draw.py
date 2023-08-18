import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]

plt.plot(x, y)
plt.xlabel('X轴',  fontsize=16)
plt.ylabel('Y轴',  fontsize=16)
plt.title('折线图', fontsize=18)
plt.show()