# operators/dummy_decay.py
class DummyDecayOperator:
    def __init__(self, lambda_decay):
        self.lambda_decay = lambda_decay

    def apply(self, state, dt):
        state.dY += -self.lambda_decay * state.Y
