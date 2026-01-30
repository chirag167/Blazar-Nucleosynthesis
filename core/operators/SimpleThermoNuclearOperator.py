# operators/thermonuclear_simple.py
class SimpleThermonuclearOperator:
    def __init__(self, reactant_i, reactant_j, product_k, rate):
        self.i = reactant_i
        self.j = reactant_j
        self.k = product_k
        self.rate = rate

    def apply(self, state, dt):
        Yi = state.Y[self.i]
        Yj = state.Y[self.j]

        dR = self.rate * Yi * Yj

        state.dY[self.i] -= dR
        state.dY[self.j] -= dR
        state.dY[self.k] += dR

