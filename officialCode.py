import os
import random
import re

import astropy.time
#import emcee
import numpy as np
import pandas as pd
import pylab

from subprocess import call
from scipy.special import sph_harm

# load configurations
from config import *


###############################################################################
#This code allows for masking of certain spans of data for specifiable pulsars#
#Time dependent ISM gradients for specifiable pulsars                         #
#It uses the true positions of the Earth and Sun                              #
#Allows for N_GRID = 1                                                         #
###############################################################################

def normalize(df):
    """row normalized version of df"""
    return (df ** 2).sum(axis=1) ** .5


# All positions measured relative to the Sun and in AU
def rho_func(dfCurrent, l, m):
    """note: this returns an np.ndarray, not a pd.DataFrame, because df.real is
    not supported in the same way that other complex number operators are.

    """
    r = normalize(dfCurrent)
    rhat = dfCurrent.divide(r, axis=0)
    phi = np.arctan(rhat.x1 / rhat.x0)
    theta = np.arccos(rhat.x2)

    if m == 0:
        val = sph_harm(m, l, phi, theta)
    elif m < 0:
        val = (1j / np.sqrt(2)) * (
            sph_harm(m, l, phi, theta) - (-1) ** m * sph_harm(-m, l, phi, theta)
        )
    elif m > 0:
        val = (1 / np.sqrt(2)) * (
            sph_harm(-m, l, phi, theta) + (-1) ** m * sph_harm(m, l, phi, theta)
        )
    val /= r ** 2

    return val.real


def gamma_func(dfEarth, dfPulsar, l, m, delta=1e-2, dMax=1e2):
    """delta and dMax are integration resolution and extent parameters"""
    k = dfPulsar - dfEarth
    kNorm = normalize(k)
    kHat = k.divide(kNorm, axis=0)

    dfCurrent = dfEarth.copy()
    dfCurrentDensity = rho_func(dfCurrent, l, m)
    dfTraversed = dfCurrent - dfEarth

    # TODO: finish this

    # a record in dfCurrent is a 3 coordinate location at a given time. we need
    # to integrate *for each time*, meaning we need to sub-set our df based on
    # the norm condition every step
    dfGamma = pd.DataFrame(data={'gamma': 0}, index=dfCurrent.index)

    while True:
        # subset all datasets based on our traversed condition
        dfTraversedNorm = normalize(dfTraversed)
        finished = (dfTraversedNorm >= kNorm) | (dfTraversedNorm >= dMax)
        unfinished = ~finished

        if finished.all():
            break
        else:
            dfGamma.loc[unfinished, 'gamma'] += delta * dfCurrentDensity[unfinished]
            dfCurrent += delta * kHat
            dfCurrentDensity = rho_func(dfCurrent, l, m)
            dfTraversed = dfCurrent - dfEarth

    #val /= pylab.norm(traversed)

    return dfGamma


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


def gen_gamma(pulsar, l, m, dense=False):
    print(pulsar, l, m)

    Lambda, Beta, Cobeta = retrieve_coordinates(pulsar)

    data = get_sorted_data(pulsar, dense)

    ssbPulsar = 1e9 * np.array([
        np.sin(Cobeta) * np.cos(Lambda),
        np.sin(Cobeta) * np.sin(Lambda),
        np.cos(Cobeta)
    ])

    ssbEarth = data[['ssb0', 'ssb1', 'ssb2']]
    sunEarth = data[['sun0', 'sun1', 'sun2']]

    if dense:
        ssbEarth *= AU_PER_LS
        sunEarth *= AU_PER_LS

    # need to rename so we can add and subtract in place
    ssbEarth.columns = sunEarth.columns = ['x0', 'x1', 'x2']

    ssbSun = ssbEarth - sunEarth
    sunPulsar = ssbSun + ssbPulsar

    dfGamma = gamma_func(sunEarth, sunPulsar, l, m)

    # add times, reorder, write to file, and return
    dfGamma.loc[:, 'time'] = data.time

    save_gammas(dfGamma, pulsar, l, m, dense, version='new')

    dfGamma.loc[:, 'pulsar'] = pulsar
    dfGamma.loc[:, 'l'] = l
    dfGamma.loc[:, 'm'] = m
    dfGamma.loc[:, 'dense'] = dense

    return dfGamma


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


def phi_hat_projector(dfEarth, ssbPulsar):
    pHat = ssbPulsar / pylab.norm(ssbPulsar)
    phi = np.arctan(pHat[1] / pHat[0])
    phiHat = np.array([-np.sin(phi), np.cos(phi), 0])
    return dfEarth.dot(phiHat)


def theta_hat_projector(dfEarth, ssbPulsar):
    pHat = ssbPulsar / pylab.norm(ssbPulsar)
    phi = np.arctan(pHat[1] / pHat[0])
    theta = np.arccos(pHat[2])
    thetaHat = np.array([
        -np.cos(phi) * np.cos(theta),
        -np.sin(phi) * np.cos(theta),
        np.sin(theta)
    ])
    return dfEarth.dot(thetaHat)


def gen_projections(pulsar, dense=False):
    Lambda, Beta, Cobeta = retrieve_coordinates(pulsar)

    data = get_sorted_data(pulsar, dense)

    ssbPulsar = 1e9 * np.array([
        np.sin(Cobeta) * np.cos(Lambda),
        np.sin(Cobeta) * np.sin(Lambda),
        np.cos(Cobeta)
    ])

    ssbEarth = data[['ssb0', 'ssb1', 'ssb2']]

    if dense:
        ssbEarth *= AU_PER_LS

    dfAngles = pd.DataFrame(
        data={
            'theta': theta_hat_projector(ssbEarth, ssbPulsar),
            'phi': phi_hat_projector(ssbEarth, ssbPulsar),
        }
    )

    dfAngles.loc[:, 'time'] = data.time.copy()
    dfAngles.loc[:, 'pulsar'] = pulsar

    return dfAngles


def window(t, lb, ub):
    return int(lb < t < ub)


def timescale_boxcar(df, xname='time', yname='dm', errname='error',
                     tau=T_SMOOTH):
    """a dataframe with x column xname, y column yname, error column errname,
    and also a tau value (the width of the boxcar window)

    """
    z = df[[xname, yname, errname, 'pulsar']].copy()

    # calculate the weights from the error
    z.loc[:, 'wt'] = z[errname] ** -2
    z.loc[:, 'wtd_y'] = z[yname] * z.wt

    # i can't use windows because we can't center on a timeseries and we can't
    # calculate ragged windows (non-standard number of records) with
    # non-timeseries. woopsies.
    def wtd_boxcar_avg(row):
        x = z[
            (z.pulsar == row.pulsar)
            & ((z[xname] - row[xname]).abs() <= tau / 2)
        ]
        return x.wtd_y.sum() / x.wt.sum()

    return z[[xname, 'pulsar', 'wt', 'wtd_y']].apply(wtd_boxcar_avg, axis=1)
    y_smoothed = []
    for i in range(len(x)):
        v1 = 0
        v2 = 0
        tpivot = x[i]
        for j in range(len(x)):
            if abs(tpivot - x[j]) <= tau / 2:
                v1 += y[j] / s[j] ** 2
                v2 += 1.0 / s[j] ** 2
        y_smoothed.append(v1 / v2)
    y_smoothed = np.array(y_smoothed)
    return y_smoothed


def lms(lmax=L_MAX):
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            yield (l, m)


def add_grid_value(df, gridCol, valCol, grid, deltaGrid):
    for (k, (gridLeft, gridRight)) in enumerate(zip(grid[:-1], grid[1:])):
        inGrid = df[df[gridCol].between(gridLeft, gridRight)]
        kLeft = '{}_{}'.format(valCol, k)
        kRight = '{}_{}'.format(valCol, k + 1)
        df.loc[inGrid.index, kLeft] = inGrid[valCol] * (
            1 - (inGrid[gridCol] - gridLeft) / deltaGrid
        )
        df.loc[inGrid.index, kRight] = inGrid[valCol] * (
            1 - (gridRight - inGrid[gridCol]) / deltaGrid
        )


###########################
# Compute Basis Functions #
###########################

if __name__ == '__main__':

    data = load_pulsar_data(pulsarsToInclude)

    pulsarGammas = pd.concat(
        objs=[
            (
                gen_gamma(pulsar, l, m, dense=False)
                if doIntegrate
                else load_gammas(pulsar, l, m, dense=False, version='new')
            )
            for pulsar in pulsarsToInclude
            for (l, m) in lms()
        ],
        ignore_index=True
    )

    angles = pd.concat(
        objs=[
            gen_projections(pulsar, dense=False) for pulsar in pulsarsToInclude
        ],
        ignore_index=True
    )

    dataDense = load_pulsar_data(pulsarsToInclude, dense=True)

    pulsarGammasDense = pd.concat(
        objs=[
            (
                gen_gamma(pulsar, l, m, dense=True)
                if doIntegrate
                else load_gammas(pulsar, l, m, dense=True, version='new')
            )
            for pulsar in pulsarsToInclude
            for (l, m) in lms()
        ],
        ignore_index=True
    )

    anglesDense = pd.concat(
        objs=[
            gen_projections(pulsar, dense=True) for pulsar in pulsarsToInclude
        ],
        ignore_index=True
    )


    ####################################
    # Apply any masks and do smoothing #
    ####################################

    # values are masked if they are between two values (a lower and upper bound)
    # specified in MASKS in the preamble above. The following will work for any
    # dataframe which has columns 'pulsar' and 'time'
    for df in [data, pulsarGammas, angles]:
        df.loc[:, 'masked'] = False
        for (pulsar, lb, ub) in MASKS:
            df.loc[
                (df.pulsar == pulsar) & df.time.between(lb, ub), 'masked'
            ] = True

    # create a smoothed version of the unmasked values
    data.loc[~data.masked, 'dm_smooth'] = timescale_boxcar(
        df=data[~data.masked],
        xname='time',
        yname='dm',
        errname='error',
        tau=T_SMOOTH
    )

    ###########################
    # Collect all coordinates #
    ###########################
    coords = pd.DataFrame(
        [retrieve_coordinates(pulsar) for pulsar in pulsarsToInclude],
        columns=['lambda', 'beta', 'cobeta']
    )

    #########################
    # Construct Interp Grid #
    #########################
    tMin = data.time.min()
    tMax = data.time.max()
    tauGrid = np.linspace(tMin, tMax, N_GRID)
    deltaTau = (tMax - tMin) / (N_GRID - 1)

    # interpolate values to grid points we just calcd
    add_grid_value(
        df=pulsarGammas, gridCol='time', valCol='gamma', grid=tauGrid,
        deltaGrid=deltaTau
    )
    add_grid_value(
        df=angles, gridCol='time', valCol='phi', grid=tauGrid,
        deltaGrid=deltaTau
    )
    add_grid_value(
        df=angles, gridCol='time', valCol='theta', grid=tauGrid,
        deltaGrid=deltaTau
    )

    ######################
    # Make Design Matrix #
    ######################

    nRow = (~data.masked).sum()

    tdgDict = dict(tDependentGradients)
    nCol = (
        len(list(lms())) * N_GRID  # N_GRID spots for each l, m pair
        + sum([
            1 + 2 * tdgDict.get(pulsar, 1)
            for pulsar in pulsarsToInclude
        ])  # dependent gradients, not sure what they are
    )

    M = np.zeros([nRow, nCol])

    # TODO: create the matrix which is just a record for each (pulsar, time)
    # pair with columns kind of all over the place (look for subsetting in the
    # actual use of this matrix in case it's overkill to form it and then break
    # it up)

    ############################
    # Make Dense Design Matrix #
    ############################

    nRowDense = (~dataDense.masked).sum()

    Mdense = np.zeros([nRowDense, nCol])


    ##############
    # Do fitting #
    ##############

    stackedHipass = []
    for i in range(len(maskedDMs)):
        for j in range(len(maskedDMs[i])):
            stackedHipass.append(maskedDMs[i][j] - maskedSmoothed[i][j])

    C = zeros([len(stackedHipass), len(stackedHipass)])
    Cinv = zeros([len(stackedHipass), len(stackedHipass)])
    index = 0
    for i in range(len(maskedErrors)):
        for j in range(len(maskedErrors[i])):
            C[index][index] = maskedErrors[i][j] ** 2
            Cinv[index][index] = maskedErrors[i][j] **  - 2
            index += 1

    CpInv = np.dot(transpose(M), np.dot(Cinv, M))
    Cp = linalg.inv(CpInv)

    deltaP = np.dot(Cp, np.dot(transpose(M), np.dot(Cinv, stackedHipass)))


    #########################################
    # Form best fit models and dense models #
    #########################################

    models = []
    modelsDense = []
    sunModelsDense = []
    pxModelsDense = []
    sunModelMoments = []
    shift = 0
    shiftDense = 0
    for i in range(len(pulsarsToInclude)):
        models.append([])
        modelsDense.append([])
        sunModelsDense.append([])
        pxModelsDense.append([])
        sunModelMoments.append([])
        for j in range(Nmoment):
            sunModelMoments[i].append([])
            for j in range(len(maskedTimes[i])):
                val = 0
                for k in range(len(deltaP)):
                    val += deltaP[k] * M[j + shift][k]
                    models[i].append(val)
                    shift += len(maskedTimes[i])
                    for j in range(len(denseTimes[i])):
                        val = 0
                        sunval = 0
                        pxval = 0
                        momentVals = zeros(Nmoment)
                        for k in range(len(deltaP)):
                            val += deltaP[k] * Mdense[j + shiftDense][k]
                            if k < N_GRID * Nmoment:
                                sunval += deltaP[k] * Mdense[j + shiftDense][k]
                                momentVals[k / N_GRID] += deltaP[k] * Mdense[j + shiftDense][k]
    else:
        pxval += deltaP[k] * Mdense[j + shiftDense][k]
        modelsDense[i].append(val)
        sunModelsDense[i].append(sunval)
        pxModelsDense[i].append(pxval)
        for k in range(Nmoment):
            sunModelMoments[i][k].append(momentVals[k])
            shiftDense += len(denseTimes[i])

    #####################
    # Compute residuals #
    #####################

    residuals = []
    for i in range(len(pulsarsToInclude)):
        residuals.append([])
        for j in range(len(models[i])):
            residuals[i].append(maskedDMs[i][j] - maskedSmoothed[i][j] - models[i][j])


    ################
    # Compute a X2 #
    ################

    X2 = 0
    nDOF =  - len(deltaP)
    for i in range(len(residuals)):
        for j in range(len(residuals[i])):
            X2 += (residuals[i][j] / maskedErrors[i][j]) ** 2
            nDOF += 1

    print(X2, nDOF, X2 / nDOF)

    ##############
    # Make plots #
    ##############


    def makePlot(index):
        pulsar = pulsarsToInclude[index]
        beta = Betas[index]

        fig = figure(figsize=(15, 12))
        #ax1 = fig.add_axes([.15, .79, .8, .15])
        ax2 = fig.add_axes([.1, .27, .85, .70])
        ax3 = fig.add_axes([.1, .10, .85, .15])
        #ax1.set_xticks([])
        ax2.set_xticks([])

        ax2.set_title(pulsar + '  ' + r'$(\beta=$' + str(beta * 180 / pi)[0:5] + r'$^\circ)$', size=18)


        ax2.set_ylabel(r'${\rm DMX}$' + ' ' + r'$[{\rm pc}$' + ' ' + r'${\rm cm}^{ - 3}]$', size=18)
        ax3.set_ylabel(r'${\rm Resids.}$', size=18)
        ax3.set_xlabel(r'${\rm Time}$' + ' ' + r'$[{\rm MJD}]$', size=18)

        #ax1.plot(denseTimes[index], modelsDense[index], 'r')
        #ax1.plot(denseTimes[index], pxModelsDense[index], 'g')
        #ax1.plot(denseTimes[index], sunModelsDense[index], 'm')

        ax2.plot(maskedTimes[index], maskedSmoothed[index], 'b')
        ax2.errorbar(pulsarTimes[index], pulsarDMs[index], pulsarErrors[index], fmt='k.')
        ax2.plot(maskedTimes[index], models[index] + maskedSmoothed[index], 'r')
        ax2.plot(maskedTimes[index], models[index] + maskedSmoothed[index], 'r.', markersize=8)

        ax3.errorbar(maskedTimes[index], residuals[index], maskedErrors[index], fmt='k.')
        ax3.plot([tMin, tMax], [0, 0], 'k')

        #ax1.set_xlim([tMin, tMax])
        ax2.set_xlim([tMin, tMax])
        ax3.set_xlim([tMin, tMax])

        fig.savefig('officialDMPlots/' + pulsar + '.png')
        close()


    if doPlot == 0:
        for i in range(len(pulsarsToInclude)):
            makePlot(i)
