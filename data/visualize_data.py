# -*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd
import optparse
import pickle


def load_db(data_folder):
    ''' load db file (pickled) to a dataframe object

    :param data_folder: str, folder path
    :return: dataframe
    '''
    db_fpath = os.path.join(data_folder, 'db.pickle')
    if not os.path.exists(db_fpath):
        raise FileNotFoundError('input database file not found.')

    db = pickle.load(open(db_fpath, 'rb'))

    data = []
    for _data, _name in db:
        data.append(_data)

    df = pd.DataFrame(np.vstack(data))

    return df


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    parser = optparse.OptionParser()
    parser.add_option(
        '-i', '--input_folder',
        help="train or evaluate",
        type='str', default='/path/to/dataset')
    opts, args = parser.parse_args()

    if not os.path.exists(opts.input_folder):
        raise FileNotFoundError('input folder not found.')

    df = load_db(opts.input_folder)
    print(80 * '-')
    print(df.describe())
    print(80 * '-')
    df.plot()
    plt.show()
