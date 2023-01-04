# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

def load_data(filepath: str):
    with open(filepath, 'r') as fin:
        line = fin.readline()
        return list(map(lambda line: list(map(float, line.strip().split(" "))), fin.readlines()))

def load_solution(filepath: str):
    with open(filepath, 'r') as fin:
        return list(map(int, fin.readline().strip().split(" ")))

if __name__ == "__main__":
    
    data = load_data("../data/image.csv")
    solution = load_solution("../solution_split_lkh.tsp")
    
    xs = list(map(lambda idx: data[idx][0], solution))
    ys = list(map(lambda idx: data[idx][1], solution))

    plt.plot(xs, ys, 'b-', lw=0.2)
    plt.savefig("solution.png", dpi=500)
