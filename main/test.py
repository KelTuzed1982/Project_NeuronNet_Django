import sys
sys.path.append( 'D:\djNeuronNet\main' )

from NeuronNet_djang import NeuronNet

'''
print 'Net.calc( [90] ) = ' + str( Net.calc( [90] ) ) 
print 'Net.calc( [30] ) = ' + str( Net.calc( [30] ) )
print 'Net.calc( [0] ) = ' + str( Net.calc( [0] ) )
Net.learn( 'learn.txt' )
print '==========IS LEARNED==========='
print 'Net.calc( [90] ) = ' + str( Net.calc( [90] ) ) 
print 'Net.calc( [30] ) = ' + str( Net.calc( [30] ) )
print 'Net.calc( [0] ) = ' + str( Net.calc( [0] ) )
#'''
'''
Net.show()
Net.learn( 'learn.txt' )
print '==========IS LEARNED==========='
Net.show()
#'''

print '==========DJANGO========='
'''
Net = NeuronNet( cntLayer = 3, nrsInLayer = [ 10, 5, 1 ], inputs = 1 )
print 'Net.calc( [90] ) = ' + str( Net.calc( [90] ) ) 
print 'Net.calc( [30] ) = ' + str( Net.calc( [30] ) )
print 'Net.calc( [0] ) = ' + str( Net.calc( [0] ) )
Net.saveDB()
print 'Id in DB = ' + str( Net.getId() )
#'''

#'''
Net = NeuronNet.loadDB( id = 5 )
print 'Net.calc( [90] ) = ' + str( Net.calc( [90] ) ) 
print 'Net.calc( [30] ) = ' + str( Net.calc( [30] ) )
print 'Net.calc( [0] ) = ' + str( Net.calc( [0] ) )
#'''

