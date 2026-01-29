# engine.py
class ReactionNetworkEngine:
    def __init__(self, state, operators):
        self.state = state
        self.operators = operators

    def step(self):
        self.state.reset_derivatives()

        for op in self.operators:
            op.apply(self.state, None)

        dt = self.state.compute_dt()
        self.state.apply_update(dt)
        self.state.t += dt

        return dt

    def run(self, t_end):
        while self.state.t < t_end:
            self.step()