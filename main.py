from fuzzy import *
from svm import *
from dtree import *
from mlp import *
import timeit
import matplotlib.pyplot as plt




if __name__ == "__main__":
    accuracy=[]
    memory_used=[]
    time=[]
    algos=["SVM","MLP","DTree"]
    fuzzy()

    accuracy.append(svm_algo())
    time.append(timeit.repeat(repeat=1))
    memory_used.append(memory_usage_psutil())
    accuracy.append(mlp())
    time.append(timeit.repeat(repeat=1))
    memory_used.append(memory_usage_psutil())
    accuracy.append(dtree())
    time.append(timeit.repeat(repeat=1))
    memory_used.append(memory_usage_psutil())

    fig=plt.figure()
    plt.title("Execution Time Vs Algorithm")
    plt.plot(algos,time)
    plt.xlabel('Algorithm')
    plt.ylabel('Execution Time in sec')
    plt.show(fig)

    fig1 = plt.figure()
    plt.title("Memory Vs Algorithm")
    plt.plot(algos, memory_used)
    plt.xlabel('Algorithm')
    plt.ylabel('Memory used in MiB')
    plt.show(fig1)

    fig2 = plt.figure()
    plt.title("Accuracy Vs Algorithm")
    plt.plot(algos, accuracy)
    plt.xlabel('Algorithm')
    plt.ylabel('Accuracy')
    plt.show(fig2)

