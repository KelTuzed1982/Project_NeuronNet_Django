from django.db import models

import cPickle
import math

#1) не забываем, что из базы строки приходят в юникоде (т.е. когда загрузим эксзепляр из базы), 
#поэтому используем encode для строковых атрибутов;
class DB_Neuron( models.Model ):
 
   __weight  = models.TextField() #массив (в pickle формате) со значениями весов (сколько элементов в массиве - столько и дендритов)
   __outFunc = models.TextField() #ссылка на функцию (в pickle формате) для выходного сигнала (ну там сигмоидная например).
                           #по умолчанию у меня будет стоять гипорболический тангенс
    
   def __unicode__( self ):
     return 'id = ' + str( self.id ) + '; weight = ' + str( cPickle.loads( self.__weight.encode() ) ) + '; outFunc = ' + str( cPickle.loads( self.__outFunc.encode() ) )
#1) не забываем, что из базы строки приходят в юникоде (т.е. когда загрузим эксзепляр из базы), 
#поэтому используем encode для строковых атрибутов;
class DB_NeuronNet( models.Model ):
 
   __layer    = models.TextField() #массив (в pickle формате) содержащий слоя неронной сети
   __R        = models.IntegerField()  #скорость обучения
   __outFunc  = models.TextField() #выходная функция (в pickle формате) для всех нейронов                            
   __dervFunc = models.TextField() #производная выходной функции
    
   def __unicode__( self ):
     return 'id = ' + str( self.id )
