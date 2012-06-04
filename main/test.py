# -*- coding: utf-8 -*-
import sys

from NeuronNet_djang import NeuronNet

'''Тестируем обучение
print '==========ДО ОБУЧЕНИЯ:==========='
Net = NeuronNet( cntLayer = 3, nrsInLayer = [ 10, 5, 1 ], inputs = 1 )
print 'Net.calc( [90] ) = ' + str( Net.calc( [90] ) ) 
print 'Net.calc( [30] ) = ' + str( Net.calc( [30] ) )
print 'Net.calc( [0] ) = ' + str( Net.calc( [0] ) )
Net.learn( 'learn.txt' )
print '==========ПОСЛЕ ОБУЧЕНИЯ:==========='
print 'Net.calc( [90] ) = ' + str( Net.calc( [90] ) ) 
print 'Net.calc( [30] ) = ' + str( Net.calc( [30] ) )
print 'Net.calc( [0] ) = ' + str( Net.calc( [0] ) )
#'''


'''
print '==========Тестируем сохранение в БД========='
Net = NeuronNet( cntLayer = 4, nrsInLayer = [ 10, 10, 10, 10 ], inputs = 1 )
print 'Net.calc( [90] ) = ' + str( Net.calc( [90] ) ) 
print 'Net.calc( [30] ) = ' + str( Net.calc( [30] ) )
print 'Net.calc( [0] ) = ' + str( Net.calc( [0] ) )
Net.saveDB()
print 'Id in DB = ' + str( Net.getId() )
#'''

'''
print '==========Тестируем загрузку из БД========='
Net = NeuronNet.loadDB( id = 1 )
print 'Net.calc( [90] ) = ' + str( Net.calc( [90] ) ) 
print 'Net.calc( [30] ) = ' + str( Net.calc( [30] ) )
print 'Net.calc( [0] ) = ' + str( Net.calc( [0] ) )
#'''

