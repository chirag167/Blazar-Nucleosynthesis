import numpy as np
import matplotlib.pyplot as plt

from core.state import NetworkState
from core.engine import ReactionNetworkEngine
from core.operators.thermonuclear import ThermonuclearOperator
from core.reactions.reaction import Reaction
from core.reactions.rate_laws import constant_rate

isotopes = ["p", "d", "he3", "he4"]

# mass numbers (for conservation check)
A = [1, 2, 3, 4]

Y0 = [0.6,0.4,0.0,0.0]  # Initial abundances
# Isotopes: p1, p2, d
state = NetworkState(
    isotopes=isotopes,
    Y0=Y0,
    temperature=1.0,
    density=1.0,
    volume=1.0
)

# Define p + p → d
reactions = [

    # p + p -> d
    Reaction(
        reactants=[(0, 1), (0, 1)],
        products=[(1, 1)],
        rate_func=constant_rate(0.5),
        name="p+p->d"
    ),

    # p + d -> he3
    Reaction(
        reactants=[(0, 1), (1, 1)],
        products=[(2, 1)],
        rate_func=constant_rate(0.3),
        name="p+d->he3"
    ),

    # d + d -> he4
    Reaction(
        reactants=[(1, 1), (1, 1)],
        products=[(3, 1)],
        rate_func=constant_rate(0.1),
        name="d+d->he4"
    )
]


thermo = ThermonuclearOperator(reactions)

engine = ReactionNetworkEngine(
    state,
    [thermo],
    record_history=True
)

engine.run(t_end=10.0)
print(state.Y)

t = np.array(engine.t_history)
Y = np.array(engine.Y_history)

plt.figure()
for i, name in enumerate(isotopes):
    plt.plot(t, Y[:, i], label=name)

plt.xlabel("Time")
plt.ylabel("Abundance Y")
plt.yscale("log")
plt.legend()
plt.title("Dummy Group-1 Thermonuclear Network")
plt.show()
