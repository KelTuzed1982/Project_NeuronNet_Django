from django.db import models

import cPickle
import math

"""
Раньше была идея, сохранять каждый нейрон сети в отдельную таблицу,
но эта идея пала смертью храбрых. На данный момент просто весь массив с 
нейронами сохраняется в БД при помощи pickle формата
class DB_Neuron( models.Model ):
 
   __weight  = models.TextField() #список (в pickle формате) со значениями весов (сколько элементов в массиве - столько и дендритов)
   __outFunc = models.TextField() #ссылка на функцию (в pickle формате) для выходного сигнала (ну там сигмоидная например).
                           
    
   def __unicode__( self ):
     return 'id = ' + str( self.id ) + '; weight = ' + str( cPickle.loads( self.__weight.encode() ) ) + '; outFunc = ' + str( cPickle.loads( self.__outFunc.encode() ) )
"""     

class DB_NeuronNet( models.Model ):
   """Класс, который представляет собой джанговскую модель для
   сохранения в БД"""
 
   __layer    = models.TextField()
   """список слоев с нейронами (в pickle формате)"""
   
   __R        = models.IntegerField()  #скорость обучения
   """скорость обучения"""
   
   __outFunc  = models.TextField() #
   """ссылка на выходную функция (в pickle формате) для всех нейронов"""
   
   __dervFunc = models.TextField()
   """ссылка на производную от выходной функции (в pickle формате) для всех нейронов"""
    
   def __unicode__( self ):
     return 'id = ' + str( self.id )
