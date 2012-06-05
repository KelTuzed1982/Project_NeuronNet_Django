# -*- coding: utf-8 -*-
import sys

from NeuronNet_djang import NeuronNet

#'''Тестируем обучение
print '==========BEFORE LEARN:==========='
Net = NeuronNet( cntLayer = 2, nrsInLayer = [ 10, 1 ], inputs = 1 )
Net.show()
print 'Net.calc( [90] ) = ' + str( Net.calc( [90] ) ) 
print 'Net.calc( [30] ) = ' + str( Net.calc( [30] ) )
#print 'Net.calc( [0] ) = ' + str( Net.calc( [0] ) )
Net.learn( 'learn.txt' )
print '==========AFTER LEARN:==========='
print 'Net.calc( [90] ) = ' + str( Net.calc( [90] ) ) 
print 'Net.calc( [30] ) = ' + str( Net.calc( [30] ) )
print 'Net.calc( [0] ) = ' + str( Net.calc( [0] ) )
Net.show()
#'''


'''
print '==========Test save to DB========='
Net = NeuronNet( cntLayer = 4, nrsInLayer = [ 10, 10, 10, 10 ], inputs = 1 )
print 'Net.calc( [90] ) = ' + str( Net.calc( [90] ) ) 
print 'Net.calc( [30] ) = ' + str( Net.calc( [30] ) )
print 'Net.calc( [0] ) = ' + str( Net.calc( [0] ) )
Net.saveDB()
print 'Id in DB = ' + str( Net.getId() )
#'''

'''
print '==========Test load from DB========='
Net = NeuronNet.loadDB( id = 1 )
print 'Net.calc( [90] ) = ' + str( Net.calc( [90] ) ) 
print 'Net.calc( [30] ) = ' + str( Net.calc( [30] ) )
print 'Net.calc( [0] ) = ' + str( Net.calc( [0] ) )
#'''

