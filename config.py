# -*- coding: utf-8 -*-
"""
Небольшая настройка проекта, а именно:
1) установка pythonpath к проекту
2) установка значения переменной DJANGO_SETTINGS_MODULE
"""

#1) установка pythonpath к проекту
import sys
sys.path.append( 'D:\Repo_GitHub\Project_NeuronNet_Django' )
#

#2) установка значения переменной DJANGO_SETTINGS_MODULE
import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'settings'
#
