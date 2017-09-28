from nn_framework.op import Maximum, Const, Mean, Ones, Variable, Const, Placeholder
from nn_framework.functions import Sigmoid, LogisticLoss
from nn_framework.session import Session

sigmoid = Sigmoid
mean = Mean
ones = Ones
logistic_loss = LogisticLoss
variable = Variable
const = Const
placeholder = Placeholder


def relu(x):
  return Maximum(Const(0), x)
