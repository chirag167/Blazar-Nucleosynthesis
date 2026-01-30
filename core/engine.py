class ReactionNetworkEngine:
    def __init__(self, state, operators, record_history=False):
        self.state = state
        self.operators = operators
        self.record_history = record_history

        if record_history:
            self.t_history = []
            self.Y_history = []

    def step(self):
        self.state.reset_derivatives()

        for op in self.operators:
            op.apply(self.state, None)

        dt = self.state.compute_dt()
        self.state.apply_update(dt)
        self.state.t += dt

        if self.record_history:
            self.t_history.append(self.state.t)
            self.Y_history.append(self.state.Y.copy())

        return dt

    def run(self, t_end):
        while self.state.t < t_end:
            self.step()
