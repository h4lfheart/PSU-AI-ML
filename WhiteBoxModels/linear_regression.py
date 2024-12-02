import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

dataset = pd.read_csv("scores.csv")

def mean_squared_error(m: float, b: float):
    err = 0
    n = len(dataset)
    for i in range(n):
        x = dataset.iloc[i]["Hours"]
        y = dataset.iloc[i]["Scores"]

        err += (y - (m * x + b)) ** 2

    err /= n

    return err

def gradient_descent(m: float, b: float, learning_rate: float):
    dm = 0
    db = 0

    n = len(dataset)
    for i in range(n):
        x = dataset.iloc[i]["Hours"]
        y = dataset.iloc[i]["Scores"]

        # woah partial derivative of loss function im learning
        dm += x * (y - (m * x + b))
        db += (y - (m * x + b))

    dm *= -(2 / n)
    db *= -(2 / n)

    return m - dm * learning_rate, b - db * learning_rate

epochs = 1000
m = 0
b = 0
learning_rate = 0.0001
current_epoch = 0

x, y = dataset["Hours"], dataset["Scores"]

figure, axes = plt.subplots()
axes.scatter(x, y, color='blue')

regression_line = axes.plot(x, m * x + b, color='red')[0]

def advance_anim(frame):

    global current_epoch
    if frame < current_epoch:
        return [regression_line]


    global m, b
    m, b = gradient_descent(m, b, learning_rate)
    regression_line.set_ydata(m * x + b)

    plt.title(f"Linear Regression: Epoch {current_epoch + 1} w/ MSE {round(mean_squared_error(m, b), 2)}")
    plt.draw()

    current_epoch += 1

    return [regression_line]


ani = FuncAnimation(figure, advance_anim, frames=epochs, blit=True, interval=5)

plt.scatter(dataset["Hours"], dataset["Scores"])
plt.show()