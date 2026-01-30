from core.state import NetworkState
from core.engine import ReactionNetworkEngine
from core.operators.dummy_decay import DummyDecayOperator

# one isotope, Y=1
state = NetworkState(
    isotopes=["X"],
    Y0=[1.0],
    temperature=1.0,
    density=1.0,
    volume=1.0
)

decay = DummyDecayOperator(lambda_decay=0.5)
engine = ReactionNetworkEngine(state, [decay])

engine.run(t_end=10.0)

print(state.Y)  # should be ~exp(-0.5 * t)
