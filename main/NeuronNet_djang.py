# -*- coding: cp1251 -*-

#��������� pythonpath
import sys

sys.path.append( 'D:\Repo_GitHub\Project_NeuronNet_Django' )
#

from neuron_net.models import DB_Neuron, DB_NeuronNet

import random
import math
import cPickle
import copy

#���������� ������� (����� �������������� ��� �������� ������� �������)
#self ����� ��-�� ���� ��� ������ ������ ���������� � ������� ���� ������ �� ����
def sigmoidFunc( x ):
    return 1 / ( 1 + math.exp( -x ) )

#����������� ��� ���������������� ��������
#self ����� ��-�� ���� ��� ������ ������ ���������� � ������� ���� ������ �� ����
def dervTanh( y ):
    return ( 1 - y ) * ( 1 + y )

def dervSigmoid( y ):
    return y * ( 1 - y )


#�����, ����������� ������ �������
class Neuron:
 
    #�����������
    def __init__( self, initWeight = [ random.uniform( -0.05, 0.05 ) ], initOutFunc = math.tanh ):
            self.__weight = initWeight
            self.__outFunc = initOutFunc        
    #
    
    #���������� ������ �����
    def getWeight( self ):
        return self.__weight
    #

    #������������� ������ �����
    def setWeight( self, newWeight ):
        self.__weight = newWeight
    #
        
    #������������� �������� �������
    def setOutFunc( self, outFunc ):
        self.__outFunc = outFunc
    #

    #������� �������� ������ �� ������� ������           
    def calc( self, inputData ):
        #������������ ����������        
        res = 0 #��������� ������������
        #
        for i, w in enumerate( self.__weight ):
            res += inputData[ i ]*w
        return self.__outFunc( res )
    #
#    

#�����, ����������� ������ ��������� ����
#� ������ ���� ������ ������ �� ���� ������ �� ����� ��������� ������������ ����
#���� ��� ������� �������� �� ���� (����� ������ ��������� � ������������)  
class NeuronNet:        

    #����������� (������� ���� ��������� �� ����)
    def __init__( self, **kwargs ):
            #TODO ����������� ���������� ��� ���� �������� �� �������
            #������������ ����������
            inputs = int() #���������� ������ � ����
            #
            
            self.__DB_Net = None #���� �� ��������� ���������� ��� ��������, � ������� ��� ���������� ������ ������
            self.__inputs = list() #������������ ������ ��� �������� ������� ������ ����� ��������� (TODO ���������)
            
            if 'R' in kwargs:
                self.__R = kwargs[ 'R' ]
            else:
                self.__R = 1
            if 'oFun' in kwargs:
                self.__outFunc = kwargs[ 'oFun' ]
            else:
                self.__outFunc = math.tanh
            if 'dervFun' in kwargs:
                self.__dervFunc = kwargs[ 'dervFun' ]
            else:
                self.__dervFunc = dervTanh
            #���������� ��� ���� ����� ������� (�� �������� ������� � ��������� ��� ������ ��������� ����������)
            if 'layer' in kwargs: #���� � ���������� ������ ����� ���� � ��������� 
                self.__layer = kwargs[ 'layer' ]
                self.__countLayer = len( self.__layer )
                
                self.__neuronsInLayer = list()
                for lay in self.__layer:
                    self.__neuronsInLayer.append( len(lay) )
            else: #����� ������� ����� ���� ��������� �� ���������� ����� � ���������� �������� � ���
                self.__layer = list()
                if 'cntLayer' in kwargs and 'nrsInLayer' in kwargs and 'inputs' in kwargs:
                    self.__countLayer = kwargs[ 'cntLayer' ]
                    self.__neuronsInLayer = kwargs[ 'nrsInLayer' ]
                    inputs = kwargs[ 'inputs' ] #����� ������ ��� �������� ����, � ��������� ������� �������� ��� �����
                else: #���� ������ �� ���� ��������� ������� �� ���������
                    self.__countLayer = 1
                    self.__neuronsInLayer = [ 1 ]
                    inputs = 2
                #������ ����
                
                lay = list() #c���            
                weight = list() #������ �����            
                SomeNeuron = object() #������            
                dendrits = int() #���������� ��������� � �������        
                #

                #������� c���
                for n_lay in range(self.__countLayer):
                    lay = list()
                    #������� ������� � ����
                    for n_neuron in range( self.__neuronsInLayer[n_lay] ):                    
                        #������� ������ � ������
                        weight = list()
                        dendrits = 0
                        if n_lay == 0: #���� ������� ���� - �� ���������� ��������� ������������ ����������� ������
                            dendrits = inputs
                        else:#����� �� ���������� ����� ���������� �������� �� ���������� ������
                            dendrits = self.__neuronsInLayer[n_lay - 1] 
                        for w in range( dendrits ):
                            weight.append( random.uniform( -0.05, 0.05 ) )
                        #
                        SomeNeuron = Neuron( weight )
                        SomeNeuron.setOutFunc( self.__outFunc )
                        lay.append( copy.deepcopy( SomeNeuron ) )
                    #                   
                    self.__layer.append(lay)
                #
    #

    #������ ������� ��� ��������� �������
    def setOutFunc( self, outFunc ): 
        self.__outFunc = outFunc
        for lay in self.__layer:
            for SomeNeuron in lay:
                SomeNeuron.setOutFunc( self.__outFunc )
    #

    #������ �������� ��������
    def setR( self, R ): 
        self.__R = R
    #

    #��������� ������������� �������
    #��� ��������� �������
    def getOutFunc( self ): 
        return self.__outFunc
    #

    #���������� id ������ � ���� ��� ������ ����
    def getId( self ):
        if ( None == self.__DB_Net  ):
            return None
        else:
            return self.__DB_Net.id

    #��������� ������������� �������� ��������
    def getR( self ): 
        return self.__R
    #
   
    def saveDB( self ):
        #������������ ����������
        pOFun  = None
        pDFun  = None
        pLayer = None
        #
        
        pOFun  = cPickle.dumps( self.__outFunc )
        pDFun  = cPickle.dumps( self.__dervFunc )
        pLayer = cPickle.dumps( self.__layer )

        
        if ( self.__DB_Net == None ): #create
            self.__DB_Net = DB_NeuronNet( _DB_NeuronNet__layer = pLayer, _DB_NeuronNet__outFunc = pOFun, _DB_NeuronNet__R = self.__R, _DB_NeuronNet__dervFunc = pDFun )
        else: #update
            self.__DB_Net._DB_NeuronNet__layer    = pLayer
            self.__DB_Net._DB_NeuronNet__outFunc  = pOFun            
            self.__DB_Net._DB_NeuronNet__R        = self.__R
            self.__DB_Net._DB_NeuronNet__dervFunc = pDFun
            
        self.__DB_Net.save() 
        
    @staticmethod
    def loadDB( *args, **kwargs ):
        #������������ ����������
        DB_Net         = object()
        Net            = object()
        DB_layer       = list()
        DB_outFunc     = None
        DB_dervFunc    = None
        DB_R           = int()    
        #
        DB_Net = DB_NeuronNet.objects.get( *args, **kwargs )
                
        DB_layer    = cPickle.loads( DB_Net._DB_NeuronNet__layer.encode() )
        DB_outFunc  = cPickle.loads( DB_Net._DB_NeuronNet__outFunc.encode() )
        DB_dervFunc = cPickle.loads( DB_Net._DB_NeuronNet__dervFunc.encode() )
        DB_R        = DB_Net._DB_NeuronNet__R
       
        Net = NeuronNet( layer = DB_layer, outFunc = DB_outFunc, R = DB_R, dervFun = DB_dervFunc )
        Net.__DB_Net = DB_Net        

        return Net 
    
       

    #��������� ��������� ����.
    #���� �������� all_y  = 1 �� ����������� ���������� ����� ������,
    #�� ����� ��������� ���������, ���� ��������
    #�����, �������� ������ ������ ���������� ����                                
    def calc( self, inputData, all_y = 0 ):
        #������������ ����������
        newInputData = list() #������, � ������ �������, ��������� �� ���� �� ��������        
        n_lay = 0 #����� ����               
        n_neuron = 0 #����� �������        
        c_neuron = 0 #���������� �������� � ����        
        SomeNeuron = object() #������        
        lay = list() #����        
        res = list() #������ ���������� ��� ������� ���� ��������
        #
        self.__inputs = inputData

        #c������ ����
        while n_lay < self.__countLayer:
             lay = self.__layer[ n_lay ]
             c_neuron = len( lay )
             newInputData = []
             n_neuron = 0
             #c������ ������ ������ � ����������� 
             while n_neuron < c_neuron:
                  SomeNeuron = lay[ n_neuron ]
                  newInputData.insert( n_neuron, SomeNeuron.calc(inputData) )
                  n_neuron += 1
             #
             n_lay += 1
             inputData = newInputData
             if all_y == 1 : 
                 res.append( newInputData )
            #
        if all_y == 0 :
            res.append( newInputData )
        return res            
    #

    #���� ��� ������ ����������� � ����.
    #��� ����� ���� ������ � ��������� ��������� ���� �������� (yMas)
    #d - ������ ���������� �������
    def __changeCoeff( self, yMas, d ):
        #������������ ����������        
        g = list() #������ � ����������� ������������� g ��� ������ ����        
        gMas = list() #������ � ����������� ������������� g ��� ���� �����        
        n_lay = 0 #����� ����        
        y = list() #������ � ��������� ��������� �������� ��� ������ ����        
        sumG = 0 #����� ������������ g �� ���� �� ����������� ����        
        SomeNeuron = object() #������        
        mas = list() #������������� ������
        lenMas = 0 #������ �������
        lenMas2 = 0 #������ �������
        neurons = list() #������ � ���������
        weightX1 = list() #���������� ������ � ������
        weightX2 = list() #��������� ������ � ������
        #c�������
        i = 0
        j = 0
        #
        wMas = list() #������ ����� �������
        #
        
        #1. ��������� g ��� ��������
        #1.1. ��������� g ��� ���������� ���� ������
        n_lay = self.__countLayer - 1
        y = yMas[ n_lay ]
        lenMas = len( y )
        while i < lenMas:
            mas.append( self.__dervFunc( y[ i ] ) * ( d[ i ] - y[ i ] ) )
            i += 1
        gMas.append( mas )       
        #1.2. ��� ������ ����� ������� �� ������� 
        n_lay -= 1
        #��� ������ ����
        while n_lay >= 0 :            
            y = yMas[ n_lay ]            
            g = gMas[ ( ( self.__countLayer - 1 ) - n_lay ) - 1 ] #����� �� ������� ����, �.�. g ��������������� �� g ����������� ����            
            neurons = self.__layer[ n_lay + 1 ] #����� ������� �� ������� ����, �.�. ���� ��� �������� g, ����� ����� ������ � ���

            #������� ��� ���� � ���� ��������� ������
            weightX2 = []
            
            i = 0
            lenMas = len( neurons )
            while i < lenMas:                            
                weightX2.append( neurons[ i ].getWeight() )
                i += 1
            #
            #������� g ��� ������� �������
            mas = []
            
            j = 0
            lenMas = len( y )
            while j < lenMas:                
                #c������ sumG ��� �������
                sumG = 0
                
                i = 0
                lenMas2 = len( g )
                while i < lenMas2:
                    sumG += g[ i ] * weightX2[ i ][ j ]
                    i += 1
                #
                
                mas.append( self.__dervFunc( y[ j ] ) * sumG )
                j += 1
            #                
            gMas.append( mas )
            n_lay -= 1
        #
        gMas.reverse() #�������������� ������, �.�. ��� � �������� �������
        #2. ������ ������ ����
        n_lay = 0 #�� ������ ������� ���� � ���������!
        y = self.__inputs
        while n_lay < ( self.__countLayer ): 
            
            mas = []            
            #��� ������� �������
            i = 0
            lenMas = len( yMas[ n_lay ] )
            while i < lenMas:
                g = gMas[ n_lay ]
                SomeNeuron = self.__layer[ n_lay ][ i ]
                wMas = SomeNeuron.getWeight()
                mas = []
                j = 0
                lenMas2 = len( wMas )
                while j < lenMas2:
                    mas.append( wMas[ j ] + self.__R * g[ i ] * y[ j ] )
                    j += 1              
                SomeNeuron.setWeight( mas )
                i += 1
            #
            y = yMas[ n_lay ] #� ����. ���� ��� ����� ��������� ��������� � ����������� ����
            n_lay += 1        
        #                        

    #������� ����, ��� ������ ����� � �������
    def learn( self, fileName = 'learn.txt' ) :
        #TODO ������� ����� �������� �� ������ � 1� ������� � �����, 
        #� ��� ����� � �����������
        #��������� ����������
        LFile = open( fileName, 'r')
        inputData = list()
        #���������� ����� �� ������� ������ 
        d = 0
        #����� ���������� �����
        y = 0 
        #c����� ��� ������������� ������
        s = list()
        #--��������� ����������
        for line in LFile:
            s = line.split()
            d = float( s.pop() )            
            inputData = []
            for el in s:
                       inputData.append( float( el ) )            
            y = self.calc( inputData, 1 )            
            self.__changeCoeff( y, [d] ) 
            #
    #        
        
    #������� ��������� ���� (����� ����, � ���� ��� ��� �������, � �������� ��� �������� �����)
    def show( self ):
        #������������ ����������
        n_lay = 0
        n_neuron = 0
        n_weight = 0
        #
        n_lay = 0       
        for lay in self.__layer:
            print '���� ' + str(n_lay) + '. �������(' + str( len(lay) ) + '):'
            n_lay += 1
            n_neuron = 0
            for SomeNeuron in lay:
                print '  ������ ' + str(n_neuron) + '. ��������(' + str( len( SomeNeuron.getWeight() ) ) + '), ����:'
                n_neuron += 1
                n_weight = 0
                for w in SomeNeuron.getWeight():
                    print '     ��� ��� �������� ' + str(n_weight) + ' = ' + str(w)
                    n_weight += 1
    #
#



