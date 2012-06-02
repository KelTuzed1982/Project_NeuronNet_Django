# -*- coding: cp1251 -*-

#добавляем pythonpath
import sys

sys.path.append( 'D:\Repo_GitHub\Project_NeuronNet_Django' )
#

from neuron_net.models import DB_Neuron, DB_NeuronNet

import random
import math
import cPickle
import copy

#Сигмоидная функция (может использоваться для выходной функции нейрона)
#self нужен из-за того что объект первым параметром в функцию сует ссылку на себя
def sigmoidFunc( x ):
    return 1 / ( 1 + math.exp( -x ) )

#Производная для гиперболического тангенса
#self нужен из-за того что объект первым параметром в функцию сует ссылку на себя
def dervTanh( y ):
    return ( 1 - y ) * ( 1 + y )

def dervSigmoid( y ):
    return y * ( 1 - y )


#Класс, имитирубщий работу нейрона
class Neuron:
 
    #конструктор
    def __init__( self, initWeight = [ random.uniform( -0.05, 0.05 ) ], initOutFunc = math.tanh ):
            self.__weight = initWeight
            self.__outFunc = initOutFunc        
    #
    
    #возвращает массив весов
    def getWeight( self ):
        return self.__weight
    #

    #устанавливает массив весов
    def setWeight( self, newWeight ):
        self.__weight = newWeight
    #
        
    #устанавливает выходную функцию
    def setOutFunc( self, outFunc ):
        self.__outFunc = outFunc
    #

    #считает выходной сигнал от входных данных           
    def calc( self, inputData ):
        #используемые переменные        
        res = 0 #результат суммирования
        #
        for i, w in enumerate( self.__weight ):
            res += inputData[ i ]*w
        return self.__outFunc( res )
    #
#    

#Класс, имитирующий работу нейронной сети
#В данной сети каждый нейрон из слоя связан со всеми нейронами вышестоящего слоя
#Слой под сетчаку выделять не надо (когда задаеш параметры в конструкторе)  
class NeuronNet:        

    #конструктор (нулевой слой учитывать не надо)
    def __init__( self, **kwargs ):
            #TODO производная задаваемая для всех нейронов по разному
            #используемые переменные
            inputs = int() #количесвто входов в сеть
            #
            
            self.__DB_Net = None #пока не произошло сохранение или загрузка, у объекта нет экземпляра модели джанги
            self.__inputs = list() #используется только для передачи входных данных между фукнциями (TODO исправить)
            
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
            #определяем как сеть будет создана (из готового массива с нейронами или просто указанием параметров)
            if 'layer' in kwargs: #если в конструтор пихают сразу слои с нейронами 
                self.__layer = kwargs[ 'layer' ]
                self.__countLayer = len( self.__layer )
                
                self.__neuronsInLayer = list()
                for lay in self.__layer:
                    self.__neuronsInLayer.append( len(lay) )
            else: #иначе смотрим какие были пожелания по количеству слоев и количеству нейронов в них
                self.__layer = list()
                if 'cntLayer' in kwargs and 'nrsInLayer' in kwargs and 'inputs' in kwargs:
                    self.__countLayer = kwargs[ 'cntLayer' ]
                    self.__neuronsInLayer = kwargs[ 'nrsInLayer' ]
                    inputs = kwargs[ 'inputs' ] #нужно только для создания сети, а постоянно хранить параметр нет нужды
                else: #если вообще не было пожеланий создаем по умолчанию
                    self.__countLayer = 1
                    self.__neuronsInLayer = [ 1 ]
                    inputs = 2
                #строим сеть
                
                lay = list() #cлой            
                weight = list() #массив весов            
                SomeNeuron = object() #нейрон            
                dendrits = int() #количество дендритов у нейрона        
                #

                #создаем cлои
                for n_lay in range(self.__countLayer):
                    lay = list()
                    #создаем нейроны в слое
                    for n_neuron in range( self.__neuronsInLayer[n_lay] ):                    
                        #создаем массив с весами
                        weight = list()
                        dendrits = 0
                        if n_lay == 0: #если нулевой слой - то количество дендритов определяется количеством входов
                            dendrits = inputs
                        else:#иначе их количество равно количеству нейронов на предыдущем уровне
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

    #Задает функцию для выходного сигнала
    def setOutFunc( self, outFunc ): 
        self.__outFunc = outFunc
        for lay in self.__layer:
            for SomeNeuron in lay:
                SomeNeuron.setOutFunc( self.__outFunc )
    #

    #Задает скорость обучения
    def setR( self, R ): 
        self.__R = R
    #

    #Возращает установленную функцию
    #для выходного сигнала
    def getOutFunc( self ): 
        return self.__outFunc
    #

    #возвращает id записи в базе для данной сети
    def getId( self ):
        if ( None == self.__DB_Net  ):
            return None
        else:
            return self.__DB_Net.id

    #Возращает установленную скорость обучения
    def getR( self ): 
        return self.__R
    #
   
    def saveDB( self ):
        #используемые переменные
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
        #используемые переменные
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
    
       

    #запускает нейронную сеть.
    #если параметр all_y  = 1 то результатом выполнения будет массив,
    #со всеми выходными сигналами, всех нейронов
    #иначе, выходной сигнал только последнего слоя                                
    def calc( self, inputData, all_y = 0 ):
        #используемые переменные
        newInputData = list() #массив, с новыми данными, получаеми из слоя от нейронов        
        n_lay = 0 #номер слоя               
        n_neuron = 0 #номер нейрона        
        c_neuron = 0 #количество нейронов в слое        
        SomeNeuron = object() #нейрон        
        lay = list() #слой        
        res = list() #массив содержащий все выходны всех нейронов
        #
        self.__inputs = inputData

        #cчитаем слой
        while n_lay < self.__countLayer:
             lay = self.__layer[ n_lay ]
             c_neuron = len( lay )
             newInputData = []
             n_neuron = 0
             #cчитаем каждый нейрон в отдельности 
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

    #один раз меняет коэффиценты в сети.
    #для этого надо массив с выходными сигналами всех нейронов (yMas)
    #d - массив правильных ответов
    def __changeCoeff( self, yMas, d ):
        #используемые переменные        
        g = list() #массив с загадочными коэффицентами g для одного слоя        
        gMas = list() #массив с загадочными коэффицентоми g для всех слоев        
        n_lay = 0 #номер слоя        
        y = list() #массив с выходными сигналами нейронов для одного слоя        
        sumG = 0 #сумма произведений g на веса на вышестоящем слое        
        SomeNeuron = object() #нейрон        
        mas = list() #промежуточный массив
        lenMas = 0 #длинна массива
        lenMas2 = 0 #длинна массива
        neurons = list() #массив с нейронами
        weightX1 = list() #одномерный массив с весами
        weightX2 = list() #двумерный массив с весами
        #cчетчики
        i = 0
        j = 0
        #
        wMas = list() #массив весов нейрона
        #
        
        #1. Посчитаем g для нейронов
        #1.1. посчитаем g для последнего слоя уровня
        n_lay = self.__countLayer - 1
        y = yMas[ n_lay ]
        lenMas = len( y )
        while i < lenMas:
            mas.append( self.__dervFunc( y[ i ] ) * ( d[ i ] - y[ i ] ) )
            i += 1
        gMas.append( mas )       
        #1.2. для других слоев считаем по другому 
        n_lay -= 1
        #для кажого слоя
        while n_lay >= 0 :            
            y = yMas[ n_lay ]            
            g = gMas[ ( ( self.__countLayer - 1 ) - n_lay ) - 1 ] #берем на уровень выше, т.к. g высчитытывается от g предыдущего слоя            
            neurons = self.__layer[ n_lay + 1 ] #берем нейроны на уровень выше, т.к. веса для текущего g, можно взять только у них

            #соберем все веса в один двумерный массив
            weightX2 = []
            
            i = 0
            lenMas = len( neurons )
            while i < lenMas:                            
                weightX2.append( neurons[ i ].getWeight() )
                i += 1
            #
            #считаем g для каждого нейрона
            mas = []
            
            j = 0
            lenMas = len( y )
            while j < lenMas:                
                #cчитаем sumG для нейрона
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
        gMas.reverse() #переворачиваем массив, т.к. шли в обратном порядке
        #2. Теперь сменим веса
        n_lay = 0 #не путать нулевой слой с сетчаткой!
        y = self.__inputs
        while n_lay < ( self.__countLayer ): 
            
            mas = []            
            #для каждого нейрона
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
            y = yMas[ n_lay ] #в след. шаге это будет выходными сигналами с предыдущего слоя
            n_lay += 1        
        #                        

    #обучает сеть, при помощи файла с данными
    def learn( self, fileName = 'learn.txt' ) :
        #TODO сделать чтобы работало не только с 1м ответом в конце, 
        #а еще чтобы с несколькими
        #локальные переменные
        LFile = open( fileName, 'r')
        inputData = list()
        #правильный ответ на входные данные 
        d = 0
        #ответ полученный сетью
        y = 0 
        #cписок для промежуточных данных
        s = list()
        #--локальные переменные
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
        
    #выводит нейронную сеть (номер слоя, к ниму все его нейроны, к нейронам все значения весов)
    def show( self ):
        #используемые переменные
        n_lay = 0
        n_neuron = 0
        n_weight = 0
        #
        n_lay = 0       
        for lay in self.__layer:
            print 'Слой ' + str(n_lay) + '. Нейроны(' + str( len(lay) ) + '):'
            n_lay += 1
            n_neuron = 0
            for SomeNeuron in lay:
                print '  Нейрон ' + str(n_neuron) + '. Дендриты(' + str( len( SomeNeuron.getWeight() ) ) + '), веса:'
                n_neuron += 1
                n_weight = 0
                for w in SomeNeuron.getWeight():
                    print '     Вес для дендрита ' + str(n_weight) + ' = ' + str(w)
                    n_weight += 1
    #
#



