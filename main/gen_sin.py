# -*- coding: cp1251 -*-
import math
import random

#������������ ����������
cntInputs = 0 #���������� ������
x = 0
x_deg = int()
s = '' #������, ������� ����� ������������ � ����
pi2 = 0
cntTests = 1000 #���������� ������
#nTest = 0 #����� �����
File = object
degreeInRadian = 180 / math.pi #���������� �������� � 1� �������
radInDeg = math.pi / 180 #���������� ������ � 1� �������
#=end- ������������ ����������

#�������������
cntInputs = 1
pi2 = 2 * math.pi
cntTests = 360
File = open( 'learn.txt', 'w' )
x_deg = 0
#=end- �������������

#����� ����� � ���������, � �� � ���������
for nTest in range( cntTests ):

    x_deg = random.randint( 0, 360 )
    #x_deg += 1
    #x_deg = int( x * degreeInRadian )
    x = x_deg * radInDeg
    
    s = ''
    for nInput in range( cntInputs ):
        s += str( x_deg ) + ' '
    s += str( math.sin( x )  )
    File.write( s + '\n' )
    
File.close()
print 'done!'
