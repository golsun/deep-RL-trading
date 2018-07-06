# -*- coding:utf-8 -*-

import os

OUTPUT_FLD = os.path.join('..', 'results')
PRICE_FLD = '/Users/xianggao/Dropbox/distributed/code_db/price coinbase/vm-w7r-db'


def makedirs(fld):
    if not os.path.exists(fld):
        os.makedirs(fld)
