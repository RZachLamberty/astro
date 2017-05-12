# configuration for the DMX analysis
import os

# file path stuff
HERE = os.path.realpath(os.path.dirname(__file__))
PLOT_DIR = os.path.join(HERE, 'officialDMPlots')

# astro constants
TREF = 55462.129
PYEAR = 365.25
AU_PER_LS = 1.0 / 499.005

# discretizing params
L_MAX = 1
N_GRID = 6
T_SMOOTH = 2.0 * PYEAR

# behavior flags
DO_INTEGRATE = False
DO_DENSE_INTEGRATE = False
DO_PLOT = False


PULSARS_TO_INCLUDE = [
    #'B1937+21',  #High DM
    #'J0340+4130',
    'J0613-0200',
    'J0645+5158',
    #'J0931-1902',  #2.6 year data span
    'J1012+5307',
    'J1024-0719',
    'J1455-3330',
    'J1600-3053',
    'J1614-2230',
    'J1643-1224',  #Intervening H2 region
    'J1713+0747',  #Extreme Scattering event
    'J1744-1134',
    #'J1747-4036',  #High DM
    #'J1832-0836',  #2.6 year Data Span. Red DMX spectrum --> spectral leakage
    'J1909-3744',
    'J1918-0642',
    'J2010-1323',
    'J2145-0750',
    #'J2302+4442',
]

#more complicated DM parallax model
T_DEPENDENT_GRADIENTS = {
    #'B1937+21': 2,
    'J0340+4130': 2,
    'J0613-0200': 2,
    'J0645+5158': 2,
    #'J0931-1902': 2,
    'J1012+5307': 2,
    'J1024-0719': 2,
    'J1455-3330': 2,
    'J1600-3053': 2,
    'J1614-2230': 2,
    'J1643-1224': 5,
    'J1713+0747': 2,
    'J1744-1134': 2,
    'J1747-4036': 2,
    #'J1832-0836': 2,
    'J1909-3744': 2,
    'J1918-0642': 2,
    'J2010-1323': 2,
    'J2145-0750': 2,
    'J2302+4442': 2,
}

#excise a certain portion of a particular data set
MASKS = [
    ('J1713+0747', 54710, 55080),
    #('J1614-2230', 54700, 54750),
    #('J1744-1134', 53200, 53400),
    #('J1909-3744', 53100, 53700),
    #('J1918-0642', 53150, 53600),
    #('J2145-0750', 53000, 53600),
]
