
from custom_nn import MLP

def main():
    model = MLP(784, 10, [128,128])
    model.print_graph()

if __name__ == "__main__":
    main()