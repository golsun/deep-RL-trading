# -*- coding:utf-8 -*-

import os

OUTPUT_FLD = os.path.join('..', 'results')
PRICE_FLD = '/Users/xianggao/Dropbox/distributed/code_db/price coinbase/vm-w7r-db'


#
def mkdir_p(folder):
    ''' make a folder in file system '''
    try:
        os.mkdir(folder)
    except OSError as exc:  # Python >2.5
        if os.path.isdir(folder):
            pass
        else:
            raise
