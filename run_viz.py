from markov_tools import analyze
from examples import example_MM1
from visualize import plot_pipeline
import matplotlib.pyplot as plt


A, d = example_MM1()

res = analyze(A, d, verbose=False)

plot_pipeline(res, savepath="pineline_MM1.png")

plt.show()
