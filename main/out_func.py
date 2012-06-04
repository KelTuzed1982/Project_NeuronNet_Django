# -*- coding: utf-8 -*-

"""
Хранит описание выходных функций для нейрона и их производных
"""

def sigmoidFunc( x ):
  """
    Возвращает результат сигмоидной функции от аргумента

    Args:
        x( float ): аргумент

    Returns:
        float.
  """
  return 1 / ( 1 + math.exp( -x ) )

def dervTanh( y ):
  """
    Возвращает результат производной от фукнции гиперболического тангенса, 
    ( данная производная выражается через саму функцию гиперболического тангенса, 
     поэтому в качестве аргумента выстает значение фукнции гиперболического тангенса
     в некоторой точке )

    Args:
         y( float ): результат фукнции гиперболического тангенса в той точке, 
         от которой надо посчитать производную

    Returns:
        float.
  """  
  return ( 1 - y ) * ( 1 + y )

def dervSigmoid( y ):
  """
    Возвращает результат производной от сигмоидной фукнции, 
    ( данная производная выражается через саму сигмоидной функцию, 
     поэтому в качестве аргумента выстает значение сигмоидной фукнции
     в некоторой точке )

    Args:
         y( float ): результат фукнции сигмоидной фукнции, 
         от которой надо посчитать производную

    Returns:
        float.
  """ 
  return y * ( 1 - y )