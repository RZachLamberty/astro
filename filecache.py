#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module: filecache.py
Created: 2017-05-11

Description:
    manage io operations for different file types (gammas, old and new, and
    coordinates)

Usage:
    <usage>

"""

import os
import re

import numpy as np
import pandas as pd


# ----------------------------- #
#   io utils for gamma files    #
# ----------------------------- #

def save_gammas(dfGamma, pulsar, l, m, dense=False, version='new'):
    fname = gamma_fname(pulsar, l, m, dense, version)
    writeHeader = version == 'new'
    dfGamma[['time', 'gamma']].to_csv(
        fname, sep=' ', index=False, header=writeHeader
    )


def load_gammas(pulsar, l, m, dense=False, version='new'):
    try:
        fname = gamma_fname(pulsar, l, m, dense, version)
        if version == 'new':
            df = pd.read_csv(fname, sep=' ', index_col=False)
        else:
            df = pd.read_csv(
                fname, sep=' ', index_col=False, header=None,
                names=['time', 'gamma']
            )

        df.loc[:, 'pulsar'] = pulsar
        df.loc[:, 'l'] = l
        df.loc[:, 'm'] = m
        df.loc[:, 'dense'] = dense

    except pd.compat.FileNotFoundError:
        print('WARNING: file not found')
        df = pd.DataFrame()

    return df


def convert_gammas(dense):
    pat = '(?P<pulsar>[^_]+)_(?P<l>[\-\d]+)_(?P<m>[\-\d]+)\.dat'

    fdir = 'gammaArraysDense' if dense else 'gammaArrays'
    for (root, dirs, files) in os.walk(fdir):
        for filename in files:
            try:
                d = re.match(pat, filename).groupdict()
                pulsar = d['pulsar']
                l = int(d['l'])
                m = int(d['m'])

                # load old and save new
                save_gammas(
                    dfGamma=load_gammas(pulsar, l, m, dense, version='old'),
                    pulsar=pulsar, l=l, m=m, dense=dense, version='new'
                )
            except Exception as e:
                print(e)


def gamma_fname(pulsar, l, m, dense=False, version='new'):
    fdir = 'gammaArraysDense' if dense else 'gammaArrays'
    v = 'v2.' if version == 'new' else ''
    return os.path.join(
        fdir,
        '{pulsar:}_{l:}_{m:}.{v:}dat'.format(pulsar=pulsar, l=l, m=m, v=v)
    )


# ----------------------------- #
#   reading coordinate files    #
# ----------------------------- #

def retrieve_coordinates(pulsar):
    coords = []

    for (root, dirs, files) in os.walk('pars'):
        for filename in files:
            if filename.startswith(pulsar):
                with open(os.path.join(root, filename), 'r') as f:
                    t = f.read()

                Lambda = float(
                    re.search('\nLAMBDA\s+([\-\d\.D]+)', t).groups()[0]
                ) * np.pi / 180
                Beta = float(
                    re.search('\nBETA\s+([\-\d\.D]+)', t).groups()[0]
                ) * np.pi / 180
                Cobeta = np.pi / 2 - Beta

                return Lambda, Beta, Cobeta


# ----------------------------- #
#   pulsar location files       #
# ----------------------------- #

def get_sorted_data(pulsar, dense=False):
    if dense:
        fname = 'denseLocations/{}.dat'
        names = [
            'time',
            'ssb0', 'ssb1', 'ssb2',
            'sun0', 'sun1', 'sun2',
            'ignore'
        ]
        sep = ' '
    else:
        fname = 'DMXWithPosition/{}.dat'
        names = [
            'time', 'dm', 'error',
            'ssb0', 'ssb1', 'ssb2',
            'sun0', 'sun1', 'sun2',
            'ignore'
        ]
        sep = '\t'
    fname = fname.format(pulsar)
    df = pd.read_table(
        fname, sep=sep, header=None, index_col=False, names=names, prefix='col'
    )
    df.loc[:, 'pulsar'] = pulsar
    df = df.sort_values(by='time')
    return df


def load_pulsar_data(pulsars, dense=False):
    return pd.concat(
        objs=[get_sorted_data(pulsar, dense) for pulsar in pulsars],
        ignore_index=True
    )
