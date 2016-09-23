# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 12:51:59 2016

@author: eugenio
"""

## testing the system
##
##

def test_system():
    
    def compare_versions(a,b):
        a2 = tuple( map(int,a.split('.')) )
        b2 = tuple( map(int,b.split('.')) )
        if a2==b2:
            return '* Same version *',True
        elif a2>b2:
            return '*** Your version is more recent ***',False
        else:
            return '*** Your version older ***',False
    
    dc0 = {}
    dc0['python'] = '2.7.11' # 2.7.11
    dc0['numpy'] = '1.11.0' # 1.11.0
    dc0['networkx'] = '1.10'
    dc0['scipy'] = '0.17.0'
    dc0['pandas'] = '0.18.0' # 0.18.0
    
    ferr = 0
    diagnosis = {0:'All required modules are present.',
              1:'All required modules ** EXCEPT pandas ** are present. This will not affect methods and classes in threshold. Some functions in threshold_util will not work.',
              2:'CRUCIAL MODULES MISSING!',
              3:'All required external modules are present. THRESHOLD MODELS CANNOT BE FOUND.',
              4:'All required modules ** EXCEPT pandas ** are present. THRESHOLD MODELS CANNOT BE FOUND.'}
    dpresent = {}
    
    from sys import version_info
    
    dc = {}
    dc['python'] = '%s.%s.%s' % version_info[:3]
    
    ferr = 0
    
    try:
        import pandas as pd
    except ImportError:
        dpresent['pandas'] = False
        ferr = 1
    else:
        dpresent['pandas'] = pd.__version__
        
    try:
        import threshold
    except ImportError:
        dpresent['threshold'] = False
        ferr += 3
    else:
        dpresent['threshold'] = True
        
    try:
        import threshold_util
    except ImportError:
        dpresent['threshold_util'] = False
        ferr += 3
    else:
        dpresent['threshold_util'] = True
    
    try:    
        import numpy
    except ImportError:
        dpresent['numpy'] = False
        ferr = 2
    else:
        dpresent['numpy'] = numpy.__version__    
    
    try:    
        from scipy import __version__ as vers_scipy
    except ImportError:
        dpresent['scipy'] = False
        ferr = 2
    else:
        dpresent['scipy'] = vers_scipy
    
    try:
        import networkx as nx
    except ImportError:
        dpresent['networkx'] = False
        ferr = 2
    else:
        dpresent['networkx'] = nx.__version__

    
    print '\n---------'
    print '---------'
    print '{:16}{:16}{:16}'.format('MODULE','TESTED FOR','YOURS')
    lflag = []
    for x in ['numpy','scipy','networkx','pandas','threshold','threshold_util']:
        v = dpresent[x]
        if x[:3] == 'thr':
            if v == False:
                print '{:16}{}'.format(x,'************ NOT PRESENT **************')
            else:
                print '{:16}{}'.format(x,'************ present **************')
        else:
            v0 = dc0[x]
            if v == False:
                print '{:16}{}'.format(x,'************ NOT PRESENT **************')
            else:
                spux,flag = compare_versions(v,v0)
                lflag.append(flag)
                print '{:16}{:16}{:16} {}'.format(x,v0,v,spux)  
                
    print '--------'
    print '--------'
    print 'DIAGNOSIS: '
    print ' *',diagnosis[ferr]
    if len(lflag):
        if all(lflag):
            print ' * Versions of installed modules match.'
        else:
            print ' * There are version which do not match. Everything may work anyway.'
    print '--------'
    print '--------\n'

    
if __name__ == '__main__':
    
    test_system()
