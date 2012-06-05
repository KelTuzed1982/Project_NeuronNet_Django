# -*- coding: utf-8 -*-

#TODO неплохо бы добавить исключения
#TODO неплохо бы добавить errorMax для learn()

#установка pythonpath к проекту
import sys
sys.path.append( 'D:\Repo_GitHub\Project_NeuronNet_Django' )

#установка значения переменной DJANGO_SETTINGS_MODULE
import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'settings'

#в этом модуле хранятся выходные функции для нейронов
from out_func import *

#подключаем модели джанги, чтобы через них сохранять сеть в БД
from neuron_net.models import DB_NeuronNet

import random
import math
import cPickle
import copy

#константы (ну на самом деле это переменные - просто лень было делать 
#отдельный класс, который бы защищал переменные от переопределения)
"""
   Если веса для нейронов не будут указаны, то по умолчанию  будут выбиратся
   веса из диапозона, который определяется двумя константами ниже 
   ( random.uniform( WEIGHT_DEF_BEG, WEIGHT_DEF_END ))   
""" 
WEIGHT_DEF_BEG =  -0.05 
WEIGHT_DEF_END =  0.05

COUNT_EPOCH    =  100
"""Количество эпох для обучения"""


class Neuron:
    """
      Класс, имитирующий некоторые функции реального нейрона
    """
     
    def __init__( self, initWeight = [ random.uniform( WEIGHT_DEF_BEG, WEIGHT_DEF_END ) ], initOutFunc = math.tanh ):
      """
        Конструктор нейрона

        Kwargs:
             initWeight( list ): список весов нейрона 
             ( n-й элемент списка соотвествует n-му дендриту нейрона ). 
             По умолчанию задается список из 1го элемента cо случайным значением
             
             initOutFunc: ссылка на выходную функцию нейрона
             По умолчанию задается ссылка  фукнцию гиперболического тангенса
      """
      self.__weight  = initWeight
      self.__outFunc = initOutFunc            
        
    def getWeight( self ):
        """
          Возвращает копию списка весов
        """
        return copy.copy( self.__weight )    

    def setWeight( self, newWeight ):
        """
          Задает список весов
          
          Args:
            newWeight( list ): новый список весов
        """
        self.__weight = copy.copy( newWeight )
        
    def setOutFunc( self, outFunc ):
        """
          Задает выходную фукнцию нейрона (по ссылке)
          
          Args:
            outFunc: ссылка на выходную функцию
        """
        self.__outFunc = outFunc    
        
    def getOutFunc( self ):
        """
          Возвращает выходную фукнцию нейрона (ссылку)
        """
        return self.__outFunc
            
    def calc( self, inputData ):
        """
          Считает и возвращает выходной сигнал нейрона от входных сигналов
          
          Args:
            inputData: входные сигналы в виде списка 
            (n-й элемент списка - входной сигнал, идущий по n-му дендриту)
          
          Return:
            float        
        """
        sum  = 0
        i    = 0 
        cntW = len( self.__weight )
        
        while ( i < cntW ):
          sum += inputData[ i ] * self.__weight[ i ]  
          i   += 1       
            
        return self.__outFunc( sum )

class NeuronNet:        
    """
      Класс, имитирующий некоторые функции реальной нейронной сети.
      Вид нейронной сети: многослойный персептрон      
      
      Attributes:
        private __R( int ): Скорость обучения. По умолчанию имеет значение 1
        
        private __DB_Net( DB_NeuronNet ): ссылка на экземпляр модели джанги, 
        используемый  при сохранении/загрузки сети в/из БД. 
        Пока не произойдет сохранение в БД, имеет значение None. 
        Ну а если сеть загрузили из БД, то сразу имеет значение
        
        private __layer( list ): двумерный список, содержащий слои с нейронами
        
        private __countLayer( int ): количество слоев сети
        
        private __neuronsInLayer( list ): список с указанием количество нейронов в слоях
        (n-й элемент списка соответвует количеству нейронов в n-м слое)
        
        private __outFunc: ссылка на выходную функцию для всех нейронов
        
        private __dervFun: ссылка на производную от выходной для всех нейронов
    """
    
    def __init__( self, **kwargs ):
            """
                Конструктор нейронной сети. 
                Нейронная сеть может быть создана разными способами, 
              в зависимости от входных параметров:
              1) передан параметр layer. Параметр должен представлять собой 
              двумерный список с уже готовыми нейронами (класс Neuron). Пример:
              layer = [ [Neuron(1)], [Neuron(2), Neuron(20)], [Neuron(3,4)] ] - будет 
              создана  сеть с 3 слоями, у которой в 1ом слое 1 нейрон с 1 дендритом 
              (вес дендрита: 1), во 2ом слое 2 нейрона, в 3ом слое 1 нейрон с 2 
              дендритами (веса дендритов: 3,4)
              2) переданы параметры nrsInLayer, inputs 
              (если хотя бы 1 из параметров не передан эффекта не возымеет). 
              nrsInLayer - список с указанием сколько нейронов 
              должно быть на каждом слое. 
              inputs - количесвто входов в сеть.
              Пример: nrsInLayer = [ 1, 2, 1 ], inputs = 2 - будет создана сеть
              с 3 слоями, где в 1ом слое: 1 нейрон с 2ми дендритами 
              (именно с 2ми, т.к. inputs = 2 - два входа в сеть), во 2ом слое: 2 нейрона,
              в 3ем слое: 1 нейрон. Весовые коэффиценты у всех нейронов случайные.
              3) по умолчанию (когда входные параметры под первые два пункта не подходят).
              Строится нейронная сеть с 1 слоем, с 1 нейроном в нем, с 2ми входами, 
              весовые коэффиценты случайные
                Остальные входные параметры не влияют на структуру нейронной сети
              
              Kwargs:
                R( float ): скорость обучения сети. 
                По умолчанию имеет значение 1
                
                oFun: ссылка на выходную функцию для всех нейронов сети. 
                (необходимо задавать вместе с параметром dervFun, иначе не возымеет действия). 
                По умолчанию - ссылка на функцию гиперболического тангенса
                
                dervFun: ссылка на производную от выходной функции для всех нейронов. 
                Производная должна быть выражена через саму выходную функцию, а 
                в качестве аргумента в нее должно передаваться значение выходной 
                функции в точке, от которой надо взять производную
                (необходимо задавать вместе с параметром oFun, иначе не возымеет действия).
                По умолчанию - ссылка на производную от функции гиперболического тангенса 
                (функция dervTanh в файле out_func.py)
                
                layer( list ): двумерный список, со слоями и с уже созданами 
                в них нейронами (подробности выше, в описании конструктора)                 
                
                nrsInLayer( list ): список, с указанием сколько нейронов должно 
                быть в каждом слое (подробности выше, в описании конструктора). 
                Необходимо задавать вместе с параметром inputs
                
                inputs( int ): количесвто входов в нейронную сеть.
                Необходимо задавать вместе с параметром inputs                
            """
            #TODO неплохо бы сделать возможность задавать для каждого нейрона свою выходную функцию и производную                      
                        
            self.__DB_Net = None
                        
            #парсим входные параметры
            if 'R' in kwargs:
                self.__R = kwargs[ 'R' ]
            else:
                self.__R = 1

            if 'oFun' in kwargs and 'dervFun' in kwargs:
                self.__outFunc  = kwargs[ 'oFun' ]
                self.__dervFunc = kwargs[ 'dervFun' ]
            else:
                self.__outFunc  = math.tanh
                self.__dervFunc = dervTanh
                            
            #определяем как сеть будет создана (из готового массива с нейронами 
            #или просто указанием количеством нейронов в слоях)
            if 'layer' in kwargs: #если в конструтор пихают сразу слои с нейронами 
                self.__layer = kwargs[ 'layer' ]
                self.__countLayer = len( self.__layer )
                
                self.__neuronsInLayer = list()
                for lay in self.__layer:
                    self.__neuronsInLayer.append( len(lay) )
            else: #иначе смотрим какие были пожелания по количеству слоев и количеству нейронов в них
                self.__layer = list()
                if 'nrsInLayer' in kwargs and 'inputs' in kwargs:
                    self.__countLayer     = len( kwargs[ 'nrsInLayer' ] )
                    self.__neuronsInLayer = kwargs[ 'nrsInLayer' ]
                    inputs = kwargs[ 'inputs' ] #нужно только для создания сети, а постоянно хранить параметр нет нужды
                else: #если вообще не было пожеланий создаем по умолчанию
                    self.__countLayer = 1
                    self.__neuronsInLayer = [ 1 ]
                    inputs = 2
                
                #строим сеть                                                                                                                  

                #создаем cлои
                for n_lay in range( self.__countLayer ):
                    lay = list() #создаем cлой 
                    
                    #создаем нейроны в слое
                    for n_neuron in range( self.__neuronsInLayer[n_lay] ):                                            
                        weight = list() #создаем массив с весами                        
                        if n_lay == 0: #если нулевой слой - то количество дендритов определяется количеством входов
                            dendrits = inputs
                        else:#иначе их количество равно количеству нейронов на предыдущем уровне
                            dendrits = self.__neuronsInLayer[n_lay - 1] 
                        for w in range( dendrits ):
                            weight.append( random.uniform( WEIGHT_DEF_BEG, WEIGHT_DEF_END ) )                        
                        SomeNeuron = Neuron( weight, self.__outFunc )                        
                        lay.append( copy.deepcopy( SomeNeuron ) )                                      
                    self.__layer.append(lay)                    
    
    def setOutFunc( self, outFunc ): 
        """
          Задает функцию для выходного сигнала для всех нейронов (по ссылке)
          
          Args:
            outFunc: Ссылка на выходную функцию для всех нейронов
        """
        self.__outFunc = outFunc
        for lay in self.__layer:
            for SomeNeuron in lay:
                SomeNeuron.setOutFunc( self.__outFunc )    
    
    def setR( self, R ): 
        """
          Задает скорость обучения
          
          Args:
            outFunc: Ссылка на выходную функцию для всех нейронов
        """
        self.__R = R    

    def getOutFunc( self ): 
        """
          Возвращает выходную функцию сети (ссылку)
        """
        return self.__outFunc    

    #возвращает id записи в базе для данной сети
    def getId( self ):
        """
          Возвращает id записи в БД для данной сети (если сеть загружена из БД, 
          или хотя бы раз сохранялась)
          
          Return:
            int
        """
        if ( None == self.__DB_Net  ):
            return None
        else:
            return self.__DB_Net.id
    
    def getR( self ):
        """
          Возращает установленную скорость обучения          
          
          Return:
            int
        """
        return self.__R
   
    def saveDB( self ):
        """
          Сохраняет сеть в БД при помощи джанги
        """
               
        pOFun  = cPickle.dumps( self.__outFunc )
        pDFun  = cPickle.dumps( self.__dervFunc )
        pLayer = cPickle.dumps( self.__layer )
        
        if ( None == self.__DB_Net  ): #если еще нет экземпляра класса джанги у сети, то надо его создать
            self.__DB_Net = DB_NeuronNet( _DB_NeuronNet__layer = pLayer, _DB_NeuronNet__outFunc = pOFun, _DB_NeuronNet__R = self.__R, _DB_NeuronNet__dervFunc = pDFun )
        else: #иначе достаточно обновить свойства
            self.__DB_Net._DB_NeuronNet__layer    = pLayer
            self.__DB_Net._DB_NeuronNet__outFunc  = pOFun            
            self.__DB_Net._DB_NeuronNet__R        = self.__R
            self.__DB_Net._DB_NeuronNet__dervFunc = pDFun
            
        self.__DB_Net.save() 
        
    @staticmethod
    def loadDB( id ):
        """
          Загружает сеть из БД при помощи джанги. В kwargs указываются поля по 
          которым надо искать сеть в БД
          
          Args:
            id( int ): номер записи нейронной сети в БД                      
        """
        DB_Net = DB_NeuronNet.objects.get( id = id )
                
        DB_layer    = cPickle.loads( DB_Net._DB_NeuronNet__layer.encode() )
        DB_outFunc  = cPickle.loads( DB_Net._DB_NeuronNet__outFunc.encode() )
        DB_dervFunc = cPickle.loads( DB_Net._DB_NeuronNet__dervFunc.encode() )
        DB_R        = DB_Net._DB_NeuronNet__R
       
        Net = NeuronNet( layer = DB_layer, outFunc = DB_outFunc, R = DB_R, dervFun = DB_dervFunc )
        Net.__DB_Net = DB_Net        

        return Net 
    
                           
    def calc( self, inputData, all_y = False ):
        """
          Считает и возрашает выходной сигнал нейронной сети          
          
          Args:
            inputData( list ): входные сигналы в виде списка
            
            all_y( bool ): если True, то будет выведены выходные сигналы, всех
            нейронов, со всех слоев в виде двумерного списка (это используется 
            во время обучения, когда нужные выходные сигналы с нескольких слоев).
            По умолчанию имеет значение False     
          
          Return:
            list
        """      
            
        res = list() #список содержащий результат фукнции
        
        #cчитаем слой
        n_lay = 0 #номер слоя
        while n_lay < self.__countLayer:
             lay = self.__layer[ n_lay ]
             c_neuron = len( lay ) #количество нейронов на слое
             newInputData = list()#массив, с выходными сигналами нейронов в слое.
              #потом эти выходные сигналы становятся входными, при переходе к следующему слою
             n_neuron = 0 #номер нейрона
             #cчитаем каждый нейрон в отдельности 
             while n_neuron < c_neuron:
                  SomeNeuron = lay[ n_neuron ]
                  newInputData.insert( n_neuron, SomeNeuron.calc( inputData ) )
                  n_neuron += 1             
             n_lay += 1
             inputData = newInputData #то что было выходными 
                #сигналами становится входными для след. слоя
             if True == all_y: #если нам нужны выходные сигналы со всех слоев, 
                #то сохраняем получившиеся выходные сигналы в результат
                 res.append( newInputData )            
        if False == all_y: #если нам нужны были выходные сигналы со всех слоев, 
          #то нужно не забыть добавить в результат последний список с выходными сигналами
            res.append( newInputData )
        return res            

    def __changeCoeff( self, yMas, d, inputs ):
        """
          Меняет весовые коэффиценты сети в зависимости 
          от полученных выходных сигналов сети и настоящего ответа
          
          Args:
            yMas( list ): полученные выходные сигналы сети, со всех слоев 
            (можно получить методом calc с аргументом all_y = True)
            
            d( list ): список ожидаемых выходных сигналов от сети (т.е. настоящий ответ)
            
            inputs( list ): список входных сигналов
        """ 

        #посчитаем g для нейронов
        gMas = list() #список со всеми коэффицентами g для всех слоев
          #заполнятся будет задом наперед, а потом перевернется
        #посчитаем g для последнего слоя
        n_lay = self.__countLayer - 1 #номер слоя
        y = yMas[ n_lay ] #выходные сигналы нейронов с одного слоя
        c_neuron = len( y ) #количество нейронов в слое
        mas = list() #промежуточный массив
        i = 0
        while i < c_neuron:
            mas.append( self.__dervFunc( y[ i ] ) * ( d[ i ] - y[ i ] ) )
            i += 1
        gMas.append( mas )       
        #для других слоев g считаем по другому 
        n_lay -= 1
        n_gMas = 0 #номер элемента списка gMas, который содержит список g 
          #коэффицентов с предыдущего слоя (чтобы высчитать g для других слоев нужны
          #g c предыдущих)
        #для кажого слоя
        while n_lay > -1:            
            y = yMas[ n_lay ]            
            g = gMas[ n_gMas ]
            neurons = self.__layer[ n_lay + 1 ] #нейроны с вышестоящего слоя, 
              #нужны для высчитывания нового g
              #(формула: g[l][j] = y[j]( 1 - y[l][j] )*sum( g[l-1]*a[l][i][j])
              
            #соберем все веса нейронов текущего слоя в один двумерный список
            weightX2 = list()
            
            i = 0
            c_gs  = len( neurons )  #количество нейронов на предыдущем слое 
            #( равно количесву коэффицентов g в предыдущем слое )
            while i < c_gs:                            
                weightX2.append( neurons[ i ].getWeight() )
                i += 1
                                        
            mas = list()            
            j = 0
            c_neuron = len( y )
            while j < c_neuron:                                
                sumG = 0 #сумма произведений g с предыдущего слоя на весовые коэффиценты
                
                i = 0
                while i < c_gs:
                    sumG += g[ i ] * weightX2[ i ][ j ]
                    i    += 1                
                
                mas.insert( j, self.__dervFunc( y[ j ] ) * sumG )
                j += 1        
            gMas.append( mas )
            n_lay  -= 1
            n_gMas += 1
        
        gMas.reverse() #переворачиваем массив, т.к. шли в обратном порядке
        
        #теперь сменим веса
        n_lay = 0
        y     = inputs
        while n_lay < ( self.__countLayer ):                   
            #для каждого нейрона
            i = 0
            c_neruon = len( yMas[ n_lay ] )
            while i < c_neruon:
                g = gMas[ n_lay ]
                SomeNeuron = self.__layer[ n_lay ][ i ]
                wMas = SomeNeuron.getWeight()
                mas = list() 
                j = 0
                c_weight = len( wMas )
                while j < c_weight:
                    mas.insert( j, wMas[ j ] + self.__R * g[ i ] * y[ j ] )
                    j += 1              
                SomeNeuron.setWeight( mas )
                i += 1            
            y = yMas[ n_lay ] #в след. шаге это будет выходными сигналами с предыдущего слоя
            n_lay += 1                           

    
    def learn( self, pathFile = 'learn.txt' ) :
        """
          Обучает нейронную сеть при помощи файла с данными          
          
          Args:
            pathFile( str ): путь к файлу, с обучающей выборкой.
            По умолчанию имеет значение 'learn.txt'                                
        """        
                        
        LFile = open( pathFile, 'r')
        inputData = list()
        
        masD = list() #список ожидаемых выходных сигналов нейронной сети (правильные ответы)         
        masIn = list() #выходной сигнал полученный сетью         
        s = list() #cписок для промежуточных данных

        #собираем обучающую выборку
        for line in LFile:
            s = line.split( chr(9) ) #ответы от входных данных разделяются табуляцией        
            masIn.append( s[ 0 ].split() ) #входные данные между собой разделяются пробелами            
            masD.append( s[ 1 ].split() ) #ответы аналогично входным данным            

        LFile.close()
        
        #преобразовываем строки в числа
        lenMas = len( masIn )        
        i = 0
        while i < lenMas:
            lenMas2 = len( masIn[ i ] )
            j = 0
            while j < lenMas2:
                masIn[ i ][ j ] = float( masIn[ i ][ j ] )
                j += 1
            i += 1 
        
        lenMas = len( masD )        
        i = 0
        while i < lenMas:
            lenMas2 = len( masD[ i ] )
            j = 0
            while j < lenMas2:
                masD[ i ][ j ] = float( masD[ i ][ j ] )
                j += 1
            i += 1 

        #запускаем обучение
        i = 0
        while i < COUNT_EPOCH: #пока не проишло COUNT_EPOCH эпох
          lenMas = len( masIn )
          j = 0
          sumD = 0
          while  j < lenMas:
              sumD += masD[ j ][ 0 ]
              yAll = self.calc( masIn[ j ], True )
              self.__changeCoeff( yAll, masD[ j ], masIn[ j ] )
              j += 1
          i += 1
          
        print 'avg = ' + str( sumD / lenMas )
                 
    def show( self ):
        """
          Выводит нейронную сеть
        """
        n_lay = 0       
        for lay in self.__layer:
            print 'Lay N ' + str(n_lay) + '(count of neurons: ' + str( len(lay) ) + '). Neurons:'
            n_lay += 1
            n_neuron = 0
            for SomeNeuron in lay:
                print '  Neuron N ' + str(n_neuron) + '(count of dendrits: ' + str( len( SomeNeuron.getWeight() ) ) + '). Dendrits:'
                n_neuron += 1
                n_weight = 0
                for w in SomeNeuron.getWeight():
                    print '     Dendrit N ' + str(n_weight) + ', weight = ' + str(w)
                    n_weight += 1
