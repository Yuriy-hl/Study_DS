from linear_algebra import Vector, dot

# Ступенчатая функция
def step_function(x):
    return 1 if x >= 0 else 0

def perceptron_output(weights: Vector, bias: float, x: Vector) -> float:
    """Возвращает 1, если персептрон 'активируется', и 0, если нет"""
    calculation = dot(weights, x) + bias
    return step_function(calculation)

and_weights = [2., 2]
and_bias = -3

assert perceptron_output(and_weights, and_bias, [1, 1]) == 1
assert perceptron_output(and_weights, and_bias, [0, 1]) == 0
assert perceptron_output(and_weights, and_bias, [1, 0]) == 0
assert perceptron_output(and_weights, and_bias, [0, 0]) == 0

or_weights = [2., 2]
or_bias = -1

assert perceptron_output(or_weights, or_bias, [1, 1]) == 1
assert perceptron_output(or_weights, or_bias, [0, 1]) == 1
assert perceptron_output(or_weights, or_bias, [1, 0]) == 1
assert perceptron_output(or_weights, or_bias, [0, 0]) == 0

not_weights = [-2.]
not_bias = 1

assert perceptron_output(not_weights, not_bias, [0]) == 1
assert perceptron_output(not_weights, not_bias, [1]) == 0

import math 

def sigmoid(t: float) -> float:
    return 1 / (1 + math.exp(-t))

def neuron_output(weights: Vector, inputs: Vector) -> float:
    # weights включает член смещения, inputs включает единицу
    return sigmoid(dot(weights,inputs))

from typing import List

def feed_forward(neural_network: List[List[Vector]],
                 input_vector: Vector) -> List[Vector]:
    """Пропускает входной вектор через нейронную сеть.
       Возвращает все слои (а не только последний)"""
    outputs: List[Vector] = []

    for layer in neural_network:
        input_with_bias = input_vector + [1] # добавляем константу
        output = [neuron_output(neuron, input_with_bias)    # Вычислить выход
                  for neuron in layer]                      # для каждого нейрона.
        outputs.append(output)                              # Сложить результаты

        # Затем вход в следующий слой является выходом этого слоя
        input_vector = output

    return outputs

xor_network = [# Скрытый слой
              [[20., 20, -30],      # Нейрон 'И'
               [20., 20, -10]],     # Нейрон 'ИЛИ'
              # Выходной слой
              [[-60., 60, -30]]]    # Нейрон '2-й вход, но не 1-й вход'

# Функция feed_forward возвращает входы всех слоев, поэтому
# [-1] получает окончательный выход и 
# [0] получает значение из результирующего вектора
assert 0.000 < feed_forward(xor_network, [0, 0])[-1][0] < 0.001
assert 0.999 < feed_forward(xor_network, [1, 0])[-1][0] < 1.000
assert 0.999 < feed_forward(xor_network, [0, 1])[-1][0] < 1.001
assert 0.000 < feed_forward(xor_network, [1, 1])[-1][0] < 0.001

def sqerror_gradients(network: List[List[Vector]],
                      input_vector: Vector,
                      target_vector: Vector) -> List[List[Vector]]:
    """С учетом нейронной сети, вектора входов и вектора целей 
       сделать предсказание и вычислить градиент потери, т.е.
       сумму квадратов ошибок по отношению к весам нейрона"""
    # Прямое прохождение 
    hidden_outputs, outputs = feed_forward(network, input_vector)

    # Градиенты по отношению к преактивационным выходам выходного нейрона
    output_deltas = [output * (1 - output) * (output - target)
                     for output, target in zip(outputs, target_vector)]
    
    # Градиенты по отношению к весам выходного нейрона
    output_grads = [[output_deltas[i] * hidden_output
                     for hidden_output in hidden_outputs + [1]]
                    for i, output_neuron in enumerate(network[-1])]

    # Градиенты по отношению к преактивационным выходам скрытого нейрона
    hidden_deltas = [hidden_output * (1 - hidden_output) *
                         dot(output_deltas, [n[i] for n in network[-1]])
                     for i, hidden_output in enumerate(hidden_outputs)]

    # Градиенты по отношению к весам скрытого нейрона
    hidden_grads = [[hidden_deltas[i] * input for input in input_vector + [1]]
                    for i, hidden_neuron in enumerate(network[0])]

    return [hidden_grads, output_grads]

import random
random.seed(0)
    
# Тренировочные данные
xs = [[0., 0], [0., 1], [1., 0], [1., 1]]
ys = [[0.], [1.], [1.], [0.]]
    
# Начать со случайных весов
network = [ # Скрытый слой: 2 выхода -> 2 выхода
            [[random.random() for _ in range(2 + 1)],   # 1-й скрытый слой
             [random.random() for _ in range(2 + 1)]],  # 2-й скрытый слой
            # Выходной слой: 2 входа -> 1 выход
            [[random.random() for _ in range(2 + 1)]]   # 1-й выхю нейрон
            ]

from gradient_descent import gradient_step
import tqdm

learning_rate = 1.0

for epoch in tqdm.trange(20000, desc="neural net for xor"):
        for x, y in zip(xs, ys):
            gradients = sqerror_gradients(network, x, y)
    
            # Сделать градиентный шаг для каждого нейрона в каждом слое
            network = [[gradient_step(neuron, grad, -learning_rate)
                        for neuron, grad in zip(layer, layer_grad)]
                       for layer, layer_grad in zip(network, gradients)]
    
# Проверить, что сеть усвоила XOR
assert feed_forward(network, [0, 0])[-1][0] < 0.01
assert feed_forward(network, [0, 1])[-1][0] > 0.99
assert feed_forward(network, [1, 0])[-1][0] > 0.99
assert feed_forward(network, [1, 1])[-1][0] < 0.01

def fizz_buzz_encode(x: int) -> Vector:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]
    
assert fizz_buzz_encode(2) == [1, 0, 0, 0]
assert fizz_buzz_encode(6) == [0, 1, 0, 0]
assert fizz_buzz_encode(10) == [0, 0, 1, 0]
assert fizz_buzz_encode(30) == [0, 0, 0, 1]


def binary_encode(x: int) -> Vector:
    binary: List[float] = []

    for i in range(10):
        binary.append(x % 2)
        x = x // 2

    return binary

xs = [binary_encode(n) for n in range(101, 1024)]
ys = [fizz_buzz_encode(n) for n in range(101, 1024)]

NUM_HIDDEN = 25

network = [
    # Скрытый слой: 10 входов -> NUM_HIDDEN выходов
    [[random.random() for _ in range(10 + 1)] for _ in range(NUM_HIDDEN)],

    # Выходной слой: NUM_HIDDEN входов -> 4 выхода
    [[random.random() for _ in range(NUM_HIDDEN + 1)] for _ in range(4)]
]

from theory.linear_algebra import squared_distance

learning_rate = 1.0

with tqdm.trange(500) as t:
        for epoch in t:
            epoch_loss = 0.0
    
            for x, y in zip(xs, ys):
                predicted = feed_forward(network, x)[-1]
                epoch_loss += squared_distance(predicted, y)
                gradients = sqerror_gradients(network, x, y)
    
                # Сделать градиентный шаг для каждого нейрона в каждом слое
                network = [[gradient_step(neuron, grad, -learning_rate)
                            for neuron, grad in zip(layer, layer_grad)]
                        for layer, layer_grad in zip(network, gradients)]
    
            t.set_description(f"fizz buzz (loss: {epoch_loss:.2f})")


def argmax(xs: list) -> int:
    """Возвращает индекс наибольшего значения"""
    return max(range(len(xs)), key=lambda i: xs[i])

assert argmax([0, -1]) == 0
assert argmax([-1, 0]) == 1
assert argmax([-1, 10, 5, 20, -3]) == 3

num_correct = 0
    
for n in range(1, 101):
    x = binary_encode(n)
    predicted = argmax(feed_forward(network, x)[-1])
    actual = argmax(fizz_buzz_encode(n))
    labels = [str(n), "fizz", "buzz", "fizzbuzz"]
    print(n, labels[predicted], labels[actual])
    
    if predicted == actual:
        num_correct += 1
    
print(num_correct, "/", 100)