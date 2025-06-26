"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import time

import minitorch


def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()

        # Submodules
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        # TODO: Implement for Task 2.5.
        middle = self.layer1.forward(x).relu()
        end = self.layer2.forward(middle).relu()
        return self.layer3.forward(end).sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        # TODO: Implement for Task 2.5.
        batch_size, _ = x.shape
        out_size = self.out_size
        x_shape_expanded = x.shape + (1,)
        x_expanded = x.view(*x_shape_expanded)
        # w_shape_expanded = (1,) + tuple(reversed(self.weights.value.shape))
        w_shape_expanded = (1,) + self.weights.value.shape
        w_expanded = self.weights.value.view(*w_shape_expanded)
        b_shape_expanded = (1,) + self.bias.value.shape
        b_expanded = self.bias.value.view(*b_shape_expanded)

        return (x_expanded * w_expanded).sum(dim=1).view(
            batch_size, out_size
        ) + b_expanded


def default_log_fn(epoch, total_loss, correct, losses, epoch_time):
    print(f"Epoch {epoch} loss {total_loss} correct {correct} time {epoch_time:.4f}s")


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            epoch_start_time = time.time()

            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            epoch_time = time.time() - epoch_start_time

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses, epoch_time)


if __name__ == "__main__":
    # PTS = 50
    # HIDDEN = 2
    # RATE = 0.5
    # data = minitorch.datasets["Simple"](PTS)
    # TensorTrain(HIDDEN).train(data, RATE)

    PTS = 50
    HIDDEN = 3
    RATE = 0.5
    data = minitorch.datasets["Diag"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
