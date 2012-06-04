# -*- coding: utf-8 -*-
"""
  Генерирует обучающую выборку для обучения функции синуса.
  Сохраняет её в файл learn.txt, в ту же директорию, где и сам этот файл gen_sin.py.
  В файле learn.txt, в каждой строчке находятся входные данные для сети и ожидаемый ответ (правильный).
  Ответ от входных данных разделяются символом табуляции (chr(9)).
  Входные данные - это градусы, а ответ - это синус от этих градусов  
"""  
  
import math
import random
   
pi2 = 2 * math.pi

cntTests = 1000
"""Количество тестов"""  

radInDeg = math.pi / 180   
"""Количество радиан в 1м градусе"""

File = open( 'learn.txt', 'w' )

for nTest in range( cntTests ):

    x_deg = random.randint( 0, 360 )
    """Аргумент x в градусах, от которого будет высчитываться синус """      
      
    x = x_deg * radInDeg
    """Аргумент x в радианах, от которого будет высчитан синус"""
    
    s = str( x_deg )
    """Строка с обучающей выборкой"""
    
    s += chr(9) + str( math.sin( x )  )
    File.write( s + '\n' )
    
File.close()
print 'Сделано!'
