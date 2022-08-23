
def a1():
    print("a1")

def a2():
    print("a2")

def a3():
    a1()
    print("a3")

def a4():
    a3()
    a2()
    print("a4")

if __name__ == '__main__':
    from pycallgraph2 import PyCallGraph
    from pycallgraph2.output import GraphvizOutput
    with PyCallGraph(output=GraphvizOutput()):
        a4()