# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from transformers import pipeline
import haha as h
import hf


def printmemain(str):
    print("[来自main]"+str)
    return

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    printmemain("我是一个函数")
    h.printme("你是什么函数")
    str = "i love you"
    analysis = hf.analysis(str)
    print(analysis)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
