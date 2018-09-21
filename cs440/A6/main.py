import numpy as np
import scaledconjugategradient.py as scg
import mlutils as ml # for draw()
from copy import copy

results = trainNNs(X, T, trainFraction, hiddenLayerStructures, numberRepetitions, numberIterations, classify)

summary = summarize(results)

best = bestNetwork(summary)

# loadtxt