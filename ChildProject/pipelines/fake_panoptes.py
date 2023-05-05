#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 13:19:22 2023

@author: lpeurey

Panoptes dummy classes in order to pass tests when using panoptes
without uploading actual chunks to zooniverse nor giving the credentials
"""
import os
import time

LOCATION_FAIL = 'rec01_52_53.mp3' # precise chunk that would fail because of bad format

TEST_MAX_SUBJECT = 5 # when testing the max subjects uploaded error, how many succeed before it occurs

class PanoptesAPIException(Exception):
    pass

class Links:    
    def add_attr(self,name,value):
        setattr(self, name, value)
        return self
        
class Subject:
    sub_number = 0
    max_subjects = 10000
    def __init__(self):
        self.id = Subject.sub_number
        Subject.sub_number += 1
        self.links = Links()
        self.links.add_attr('project','')
        self.metadata = {}
    
    def add_location(self, path):
        if LOCATION_FAIL == os.path.basename(path) : raise PanoptesAPIException('Planned exception for test purposes on an invalid Location')
        self.location = path
        
    def save(self):
        if Subject.sub_number >= Subject.max_subjects: #fails at ID 5+
            raise PanoptesAPIException('User has uploaded {} subjects of {} maximum'.format(Subject.sub_number,Subject.max_subjects))
    
    def find(num):
        s = Subject()
        s.id = num
        return s
        
class SubjectSet:
    ss_n = 0
    add_number = 0
    def __init__(self):
        self.id = SubjectSet.ss_n
        SubjectSet.ss_n += 1
        self.links = Links()
        self.links.add_attr('project','')
        self.display_name = ''
        
    def save(self):
        pass
    
    #panoptes sometimes fails with this command (Stale object)
    # so here we make it fail on a precise list of occurences
    # as well as a precise subject_set despite retries
    def add(self, subjects):
        SubjectSet.add_number += 1
        if subjects.id == 35 : raise PanoptesAPIException('Planned exception for test purposes')
        
        #Fail for some additions (two in a row as we have 3 retries)
        if SubjectSet.add_number in [2,25,26,30] : raise PanoptesAPIException('Planned exception for test purposes')

class Panoptes:
    def connect(username, password):     
        assert username == 'test_username'
        assert password == 'test_password'
        
class Workflow:
    def __init__(self):
        self.id = 'test_wf'
        self.tasks = {'test_task_id':{'answers': {'label':'test_task_answer'}}}
        
class Project:
    def __init__(self, ident):
        ss1 = SubjectSet()
        ss1.display_name = 'ss1'
        ss2 = SubjectSet()
        ss2.display_name = 'ss2'
        wf = Workflow()
        self.links = Links()
        self.links.add_attr('subject_sets',[ss1,ss2])
        self.links.add_attr('workflows',[wf])
        
class Classification:
    def where(self, scope, page_size, project_id):
        return [{'links': {'user' : 'test_user_id',
                           'subjects': ['000'],
                           'workflow': 'test_wf'},
                 'annotations': [{'task': 'test_task_id', 'value':0}],
                 'id':'test_id',
                 }]
    
def reset_tests():
    Subject.sub_number = 0
    SubjectSet.add_number = 0
        
  
