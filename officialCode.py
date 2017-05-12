import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import filecache

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

    #val /= np.linalg.norm(traversed)

    return dfGamma


def gen_gamma(pulsar, l, m, dense=False):
    print(pulsar, l, m)

    Lambda, Beta, Cobeta = filecache.retrieve_coordinates(pulsar)

    data = filecache.get_sorted_data(pulsar, dense)

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


def phi_hat_projector(dfEarth, ssbPulsar):
    pHat = ssbPulsar / np.linalg.norm(ssbPulsar)
    phi = np.arctan(pHat[1] / pHat[0])
    phiHat = np.array([-np.sin(phi), np.cos(phi), 0])
    return dfEarth.dot(phiHat)


def theta_hat_projector(dfEarth, ssbPulsar):
    pHat = ssbPulsar / np.linalg.norm(ssbPulsar)
    phi = np.arctan(pHat[1] / pHat[0])
    theta = np.arccos(pHat[2])
    thetaHat = np.array([
        -np.cos(phi) * np.cos(theta),
        -np.sin(phi) * np.cos(theta),
        np.sin(theta)
    ])
    return dfEarth.dot(thetaHat)


def gen_projections(pulsar, dense=False):
    Lambda, Beta, Cobeta = filecache.retrieve_coordinates(pulsar)

    data = filecache.get_sorted_data(pulsar, dense)

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


def smear(df, grid, gridCol, valCol, inclusive=True, subindexer=None):
    """given a dataframe df with columns gridCol (the column which grid breaks
    up into pieces) and valCol (the column whose value should be smeared onto
    grid), smear to the left and right grid points proportional to the time
    distance between those points. if inclusive is True, include the first and
    last points; if not, exclude them. this is to do some funky problematic
    backwards compatability with the way the window function was previously
    implemented.

    args:
        df (pd.DataFrame): dataframe to be smeared (updated in place)
        grid (list): a single list of gridpoints
        gridCol (str): column name to be "gridded"
        valCol (str): column name of value to smear
        inclusive (bool): whether or not to include end points of grid
            (default: True)
        subindexer (None or pd.Series): if not None, will subset df to
            df[subindexer] and apply the grid to that subset of the dataframe
            (note: will create the columns regardless, so other values will
            remain null)

    """
    if subindexer is None:
        subindexer = pd.Series(data=True, index=df.index)

    N = len(grid)
    keys = ['{}_{}_{}'.format(valCol, N, i) for i in range(len(grid))]
    for (i, (gridLeft, gridRight)) in enumerate(zip(grid[:-1], grid[1:])):
        rowsToUpdate = (
            subindexer
            & df[gridCol].between(gridLeft, gridRight, inclusive=inclusive)
        )

        delta = gridRight - gridLeft
        kLeft, kRight = keys[i: i + 2]

        vals = df.loc[rowsToUpdate, valCol]
        gridVals = df.loc[rowsToUpdate, gridCol]
        df.loc[rowsToUpdate, kLeft] = vals * (gridRight - gridVals) / delta
        df.loc[rowsToUpdate, kRight] = vals * (gridVals - gridLeft) / delta

    df.loc[subindexer, keys] = df.loc[subindexer, keys].fillna(0)


def make_design_matrix(nRow, nCol, gammas, angles, masked=False):
    """make the design matrix"""
    m = np.zeros([nRow, nCol])

    # gamma
    gammacols = sorted([
        c for c in gammas.columns if re.match('gamma_\d_\d', c)
    ])
    mGamma = pd.melt(
        gammas[~gammas.masked] if masked else gammas,
        id_vars=['pulsar', 'time', 'l', 'm'],
        value_vars=gammacols
    )
    mGamma = mGamma.set_index(['pulsar', 'time', 'l', 'm', 'variable']).sort_index()
    mGamma = mGamma.unstack(['l', 'm', 'variable'])
    mGamma = mGamma.fillna(0)

    nGamma = mGamma.shape[1]
    m[:, :nGamma] = mGamma.values

    # (theta, phi) per gradient, block diagonal by pulsar
    rowStart = 0
    for pulsar in angles.pulsar.unique():
        theseAngles = angles[(angles.pulsar == pulsar) & ~angles.masked]

        # start row is rowStart, rowStop is that plus the number of time records
        # available in the unmasked angles df
        rowStop = rowStart + theseAngles.shape[0]

        # right-most empty index in m is where columns start
        colIsZero = (m == 0).all(axis=0)
        colStart = np.where(colIsZero)[0][0]

        n = T_DEPENDENT_GRADIENTS.get(pulsar)
        for angleKey in ['theta', 'phi']:
            # get the columns
            if n is None or n == 1:
                cols = [angleKey]
            else:
                cols = sorted([
                    c for c in theseAngles.columns
                    if re.match('{}_{}_\d'.format(angleKey, n), c)
                ])

                if n != len(cols):
                    msg = "should have {} smeared vals in T_DEPENDENT_GRADIENTS"
                    msg = msg.format(n)
                    raise ValueError(msg)

            # subset the array and paste 'er in
            m[rowStart: rowStop, colStart: colStart + n] = theseAngles[cols].values

            colStart += n

        # constant term (for fitting)
        m[rowStart: rowStop, colStart] = 1

        rowStart = rowStop

    return m


def main():
    ###########################
    # Compute Basis Functions #
    ###########################

    data = filecache.load_pulsar_data(PULSARS_TO_INCLUDE)

    pulsarGammas = pd.concat(
        objs=[
            (
                gen_gamma(pulsar, l, m, dense=False)
                if DO_INTEGRATE
                else filecache.load_gammas(pulsar, l, m, dense=False)
            )
            for pulsar in PULSARS_TO_INCLUDE
            for (l, m) in lms()
        ],
        ignore_index=True
    )

    angles = pd.concat(
        objs=[
            gen_projections(pulsar, dense=False) for pulsar in PULSARS_TO_INCLUDE
        ],
        ignore_index=True
    )

    dataDense = filecache.load_pulsar_data(PULSARS_TO_INCLUDE, dense=True)

    pulsarGammasDense = pd.concat(
        objs=[
            (
                gen_gamma(pulsar, l, m, dense=True)
                if DO_DENSE_INTEGRATE
                else filecache.load_gammas(pulsar, l, m, dense=True, version='new')
            )
            for pulsar in PULSARS_TO_INCLUDE
            for (l, m) in lms()
        ],
        ignore_index=True
    )

    anglesDense = pd.concat(
        objs=[
            gen_projections(pulsar, dense=True) for pulsar in PULSARS_TO_INCLUDE
        ],
        ignore_index=True
    )


    ####################################
    # Apply any masks and do smoothing #
    ####################################

    # values are masked if they are between two values (a lower and upper bound)
    # specified in MASKS in the preamble above. The following will work for any
    # dataframe which has columns 'pulsar' and 'time'
    maskDfs = [
        data, dataDense, pulsarGammas, pulsarGammasDense, angles, anglesDense
    ]
    for df in maskDfs:
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
        [filecache.retrieve_coordinates(pulsar) for pulsar in PULSARS_TO_INCLUDE],
        columns=['lambda', 'beta', 'cobeta']
    )
    coords.loc[:, 'pulsar'] = PULSARS_TO_INCLUDE
    coords = coords.set_index('pulsar')

    #########################
    # Construct Interp Grid #
    #########################

    # add "smeared" values to gamma, theta, and phi datasets based on N_GRID and
    # the values in the T_DEPENDENT_GRADIENTS dict

    # gammas are easy
    grid = np.linspace(pulsarGammas.time.min(), pulsarGammas.time.max(), N_GRID)
    smear(
        df=pulsarGammas, grid=grid, gridCol='time', valCol='gamma',
        inclusive=False, subindexer=None
    )
    # to be consistent with pevious implementation, we use the previous time
    # grid. I think this is wrong, and once we are in agreement with the end
    # results we should undo this and see what goes wrong
    smear(
        df=pulsarGammasDense, grid=grid, gridCol='time', valCol='gamma',
        inclusive=False, subindexer=None
    )

    # phi and theta, less easy. we have to generate the grids on a per-pulsar
    # basis, and we have weird time conventions
    for n in set(T_DEPENDENT_GRADIENTS.values()):
        for valCol in ['phi', 'theta']:
            # group by pulsar and grid within that pulsar's times
            for pulsar in angles.pulsar.unique():
                subtimes = data[(data.pulsar == pulsar) & ~data.masked].time
                grid = np.linspace(subtimes.min(), subtimes.max(), n)

                subindexer = angles.pulsar == pulsar
                smear(
                    df=angles, grid=grid, gridCol='time', valCol=valCol,
                    inclusive=False, subindexer=subindexer
                )

                subindexer = anglesDense.pulsar == pulsar
                smear(
                    df=anglesDense, grid=grid, gridCol='time', valCol=valCol,
                    inclusive=True, subindexer=subindexer
                )

    ######################
    # Make Design Matrix #
    ######################

    nRow = (~data.masked).sum()
    nRowDense = dataDense.shape[0]

    N_LMS = len(list(lms()))
    N_GAMMA = N_LMS * N_GRID  # N_GRID spots for each l, m pair
    nCol = (
        N_GAMMA
        + sum([
            1 + 2 * T_DEPENDENT_GRADIENTS.get(pulsar, 1)
            for pulsar in PULSARS_TO_INCLUDE
        ])  # dependent gradients, not sure what they are
    )

    M = make_design_matrix(nRow, nCol, pulsarGammas, angles, masked=True)
    Mdense = make_design_matrix(
        nRowDense, nCol, pulsarGammasDense, anglesDense, masked=False
    )

    ##############
    # Do fitting #
    ##############

    # basic assumption here is that there exists a deltaP: M deltaP = smoothed.
    # I'm not even sure of that any more, honestly
    unmasked = data[~data.masked]
    hipass = unmasked[['pulsar', 'dm', 'dm_smooth']].copy()
    hipass.loc[:, 'dm_delta'] = hipass.dm - hipass.dm_smooth

    # make a square matrix which contains M so we can invert it? Why do the
    # error terms when they'll wash out below?
    c = np.diag(unmasked.error.values ** 2)
    cInv = np.diag(unmasked.error.values ** -2)

    cpInv = M.T @ cInv @ M
    cp = np.linalg.inv(cpInv)

    # the below ultimately resolves to: deltaP = M^-1 . smoothed, or
    # M deltaP = smoothed
    deltaP = cp @ M.T @ cInv @ hipass.dm_delta.values


    #########################################
    # Form best fit models and dense models #
    #########################################

    N_SUN = N_GRID * N_LMS
    N_MOMENT_GRID = list(range(0, N_GAMMA + 1, N_GRID))

    models = M @ deltaP
    modelsDense = Mdense @ deltaP
    sunModelsDense = Mdense[:, :N_SUN] @ deltaP[:N_SUN]
    pxModelsDense = Mdense[:, N_SUN:] @ deltaP[N_SUN:]
    sunModelMoments = np.array([
        Mdense[:, i0: i1] @ deltaP[i0: i1]
        for (i0, i1) in zip(N_MOMENT_GRID[:-1], N_MOMENT_GRID[1:])
    ]).T

    #####################
    # Compute residuals #
    #####################

    residuals = unmasked.dm - unmasked.dm_smooth - models


    ################
    # Compute a X2 #
    ################

    chi2 = ((residuals - unmasked.error) ** 2).sum()
    nDof = residuals.shape[0] - deltaP.shape[0]
    print('chi2 = {}'.format(chi2))
    print('nDof = {}'.format(nDof))
    print('chi2 / nDof = {}'.format(chi2 / nDof))

    ##############
    # Make plots #
    ##############

    def make_plot(pulsar):
        unmasked = data[~data.masked]
        beta = coords.loc[pulsar].beta

        fig = plt.figure(figsize=(15, 12))
        ax2 = fig.add_axes([.1, .27, .85, .70])
        ax3 = fig.add_axes([.1, .10, .85, .15])
        ax2.set_xticks([])

        ax2.set_title(
            '{}  $(\\beta={:.5}^\\circ)$'.format(pulsar, beta * 180 / np.pi),
            size=18
        )

        ax2.set_ylabel('${\\rm DMX}$ $[{\\rm pc}$ ${\\rm cm}^{-3}]$', size=18)
        ax3.set_ylabel('${\\rm Resids.}$', size=18)
        ax3.set_xlabel('${\\rm Time}$ $[{\\rm MJD}]$', size=18)

        pulsarInd = unmasked.pulsar == pulsar
        d = unmasked[pulsarInd]
        modInd = np.where(pulsarInd)[0]

        ax2.plot(d.time, d.dm_smooth, 'b')
        ax2.errorbar(d.time, d.dm, d.error, fmt='k.')
        ax2.plot(d.time, models[modInd] + d.dm_smooth, 'r')
        ax2.plot(d.time, models[modInd] + d.dm_smooth, 'r.', markersize=8)

        ax3.errorbar(d.time, residuals[pulsarInd], d.error, fmt='k.')
        ax3.plot([unmasked.time.min(), unmasked.time.max()], [0, 0], 'k')

        #ax1.set_xlim([unmasked.time.min(), unmasked.time.max()])
        ax2.set_xlim([unmasked.time.min(), unmasked.time.max()])
        ax3.set_xlim([unmasked.time.min(), unmasked.time.max()])

        #plt.show()
        #plt.close('all')

        if not os.path.isdir(PLOT_DIR):
            os.makedirs(PLOT_DIR)
        fig.savefig(os.path.join(PLOT_DIR, '{}.png'.format(pulsar)))
        plt.close()

    if DO_PLOT == 0:
        for pulsar in PULSARS_TO_INCLUDE:
            make_plot(pulsar)


if __name__ == '__main__':
    main()
