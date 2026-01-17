from typing import List

Vector = List[float]

height_weight_age = [175,
                     68,
                     40]

grades = [95,
          80,
          75,
          62]


def add(v: Vector, w: Vector) -> Vector:
    assert len(v) == len(w)

    return [v_i + w_i for v_i, w_i in zip(v,w)]

assert add([1,2,3],[4,5,6]) == [5,7,9]

def subtract(v: Vector, w: Vector) -> Vector:
    assert len(v) == len(w)

    return [v_i - w_i for v_i, w_i in zip(v,w)]

assert subtract([5,7,9],[4,5,6]) == [1,2,3]

def vector_sum(vectors: List[Vector]) -> Vector:
    assert vectors

    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors)

    return [sum(vector[i] for vector in vectors) 
            for i in range(num_elements)]

assert vector_sum([[1, 2], [3, 4], [5, 6],[7, 8]]) == [16, 20]

def scalar_multiply(c: float, v: Vector) -> Vector:
    return [c * v_i for v_i in v]

assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]

def vector_mean(vectors: List[Vector]) -> Vector:
    """Вычисляет поэлементное среднее арифмитическое"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]
def dot(v: Vector, w: Vector) -> float:
    assert len(v) == len(w)

    return sum(v_i * w_i for v_i, w_i in zip(v, w))

assert dot([1, 2, 3], [4, 5, 6]) == 32

def sum_of_squares(v: Vector) -> float:
    return dot(v, v)

assert sum_of_squares([1, 2, 3]) == 14

import math

def magnitude(v: Vector) -> float:
    """Возвращает магнитуду (или длину) вектора v"""
    return math.sqrt(sum_of_squares(v))

assert magnitude([3, 4]) == 5


# Квадрат расстояния между двумя векторами
def squared_distance(v: Vector, w: Vector) -> float:
    return sum_of_squares(subtract(v, w))

# Расстояние между двумя векторами
def distance(v: Vector, w: Vector) -> float:
    return math.sqrt(squared_distance(v, w))    # или return magnitude(subtract(v, w))


from typing import Callable

Matrix = List[List[float]]

def make_matrix(num_rows: int,
                num_cols: int,
                entry_fn: Callable[[int, int], float]) -> Matrix:
    """
    Returns a num_rows x num_cols matrix
    whose (i,j)-th entry is entry_fn(i, j)
    """
    return [[entry_fn(i, j)             
             for j in range(num_cols)]  
            for i in range(num_rows)]   

