import numpy as np
import matplotlib.pyplot as plt

from core.state import NetworkState
from core.engine import ReactionNetworkEngine
from core.operators.SimpleThermoNuclearOperator import SimpleThermonuclearOperator

state = NetworkState(
    isotopes=["p1", "p2", "d"],
    Y0=[0.5, 0.4, 0.0],
    temperature=1.0,
    density=1.0,
    volume=1.0
)

op = SimpleThermonuclearOperator(reactant_i=0, reactant_j=1, product_k=2, rate=1.0)

engine = ReactionNetworkEngine(state, [op], record_history=True)
engine.run(t_end=5.0)

t = np.array(engine.t_history)
Y = np.array(engine.Y_history)

plt.figure()

plt.plot(t, Y[:, 0], label="p1")
plt.plot(t, Y[:, 1], label="p2")
plt.plot(t, Y[:, 2], label="d")
plt.axhline(1.0, color='black', lw=2.0, ls='--')

plt.xlabel("Time")
plt.ylabel("Abundance Y")
plt.legend()
plt.title("Simple thermonuclear reaction: p + p → d")
plt.yscale("log")
plt.grid()
plt.show()
