from numpy import *
from pylab import *
from subprocess import call
from scipy.special import sph_harm
import emcee
import random

###############################################################################
#This code allows for masking of certain spans of data for specifiable pulsars#
#Time dependent ISM gradients for specifiable pulsars                         #
#It uses the true positions of the Earth and Sun                              #
#Allows for NGrid = 1                                                         #
###############################################################################

TREF = 55462.129
PYEAR = 365.25
AUperLS = 1.0/499.005

lmax = 1
NGrid = 6
tSmooth = 2.0*PYEAR

doIntegrate = 1
doDenseIntegrate = 1
doPlot = 0
doEmcee = 1
emceeSteps = 3000
emceeBurn = 0


pulsarsToInclude = [##'B1937+21', #High DM                                                                     
                    #'J0340+4130',
                    'J0613-0200',
                    'J0645+5158',
                    ##'J0931-1902',#2.6 year data span
                    'J1012+5307',
                    'J1024-0719',
                    'J1455-3330',
                    'J1600-3053',
                    'J1614-2230',
                    'J1643-1224', #Intervening H2 region                                                          
                    'J1713+0747', #Extreme Scattering event                                                          
                    'J1744-1134',
                    #'J1747-4036', #High DM                                                                          
                    ##'J1832-0836', #2.6 year Data Span. Red DMX spectrum --> spectral leakage                        
                    'J1909-3744',
                    'J1918-0642',
                    'J2010-1323',
                    'J2145-0750',
                    #'J2302+4442'
                    ]


#more complicated DM parallax model                                                                                        
tDependentGradients = [##('B1937+21,2),
                       ('J0340+4130',2),
                       ('J0613-0200',2),
                       ('J0645+5158',2),
                       ##('J0931-1902',2),
                       ('J1012+5307',2),
                       ('J1024-0719',2),
                       ('J1455-3330',2),
                       ('J1600-3053',2),
                       ('J1614-2230',2),
                       ('J1643-1224',5),
                       ('J1713+0747',2),
                       ('J1744-1134',2),
                       ('J1747-4036',2),
                       ##('J1832-0836',2),
                       ('J1909-3744',2),
                       ('J1918-0642',2),
                       ('J2010-1323',2),
                       ('J2145-0750',2),
                       ('J2302+4442',2)
                       ]

#excise a certain portion of a particular data set                                                                         
masks = [('J1713+0747',54710,55080),
         #('J1614-2230',54700,54750),
         #('J1744-1134',53200,53400),
         #('J1909-3744',53100,53700),
         #('J1918-0642',53150,53600),
         #('J2145-0750',53000,53600)
         ]


#All positions measured relative to the Sun and in AU
def rhoFunc(position,l,m):
    r = norm(position)
    rhat = position/r
    phi = arctan(rhat[1]/rhat[0])
    theta = arccos(rhat[2])
    if m == 0:
        val = sph_harm(m,l,phi,theta)
    elif m < 0:
        val = (1j/sqrt(2))*(sph_harm(m,l,phi,theta)-(-1)**m*sph_harm(-m,l,phi,theta))
    elif m > 0:
        val = (1/sqrt(2))*(sph_harm(-m,l,phi,theta)+(-1)**m*sph_harm(m,l,phi,theta))
    val /= r**2
    return real(val)

def gammaFunc(ePos,pPos,l,m):
    #integration resolution and extent parameters
    delta = 1e-2
    dMax = 1e2

    ePos = array(ePos)
    pPos = array(pPos)

    k = pPos-ePos
    khat = k/norm(k)

    currentPos = copy(ePos)
    currentDensity = rhoFunc(currentPos,l,m)
    traversed = currentPos - ePos
    val = 0
    while norm(traversed) < norm(k) and norm(traversed) < dMax:
        val += delta*currentDensity
        currentPos[0] += delta*khat[0]
        currentPos[1] += delta*khat[1]
        currentPos[2] += delta*khat[2]
        currentDensity = rhoFunc(currentPos,l,m)
        traversed = currentPos - ePos
    #val /= norm(traversed)
    return val

def retrieveCoordinates(pulsar):
    call('grep LAMBDA pars/'+pulsar+'* > temp.txt',shell=True)
    temp = genfromtxt('temp.txt')
    call('rm temp.txt',shell=True)
    Lambda = temp[0][1]*pi/180
    call('grep BETA pars/'+pulsar+'* > temp.txt',shell=True)
    temp = genfromtxt('temp.txt')
    Beta = temp[0][1]*pi/180
    Cobeta = pi/2-Beta
    call('rm temp.txt',shell=True)
    return Cobeta,Lambda

def getSortedData(pulsar):
    data = genfromtxt('DMXWithPosition/'+pulsar+'.dat')
    holder = []
    for i in range(len(data)):
        holder.append([data[i][0],data[i][1],data[i][2],data[i][3],data[i][4],data[i][5],data[i][6],data[i][7],data[i][8]])
    holder = array(holder)
    sortedHolder = holder[holder[:,0].argsort()]
    return sortedHolder

def getDenseSortedData(pulsar):
    data = genfromtxt('denseLocations/'+pulsar+'.dat')
    holder = []
    for i in range(len(data)):
        holder.append([data[i][0],data[i][1],data[i][2],data[i][3],data[i][4],data[i][5],data[i][6],data[i][7]])
    holder = array(holder)
    sortedHolder = holder[holder[:,0].argsort()]
    return sortedHolder

def genGamma(pulsar,l,m,dense):
    print pulsar,l,m
    Cobeta,Lambda = retrieveCoordinates(pulsar)
    if dense == 0:
        data = getDenseSortedData(pulsar)
    else:
        data = getSortedData(pulsar)
    times = data[:,0]
    vals = []
    ssb_pulsar = 1e9*array([sin(Cobeta)*cos(Lambda),sin(Cobeta)*sin(Lambda),cos(Cobeta)])
    for i in range(len(times)):
        if dense == 0:
            ssb_earth = array([data[:,1][i],data[:,2][i],data[:,3][i]])*AUperLS
            sun_earth = -array([data[:,4][i],data[:,5][i],data[:,6][i]])*AUperLS
        else:
            ssb_earth = array([data[:,3][i],data[:,4][i],data[:,5][i]])
            sun_earth = array([data[:,6][i],data[:,7][i],data[:,8][i]])
        ssb_sun = ssb_earth - sun_earth
        sun_pulsar = ssb_pulsar + ssb_sun
        vals.append(gammaFunc(sun_earth,sun_pulsar,l,m))
    if dense == 0:
        f = open('gammaArraysDense/'+pulsar+'_'+str(l)+'_'+str(m)+'.dat','w')
        for i in range(len(times)):
            f.write(str(times[i])+' '+str(vals[i])+'\n')
        f.close()
    else:
        f = open('gammaArrays/'+pulsar+'_'+str(l)+'_'+str(m)+'.dat','w')
        for i in range(len(times)):
            f.write(str(times[i])+' '+str(vals[i])+'\n')
        f.close()
    return times,vals
    

def phiHatProjector(ePos,pPos):
    pHat = pPos/norm(pPos)
    phi = arctan(pHat[1]/pHat[0])
    phiHat = array([-sin(phi),cos(phi),0])
    return dot(ePos,phiHat)

def thetaHatProjector(ePos,pPos):
    pHat = pPos/norm(pPos)
    phi = arctan(pHat[1]/pHat[0])
    theta = arccos(pHat[2])
    thetaHat = array([-cos(phi)*cos(theta),-sin(phi)*cos(theta),sin(theta)])
    return dot(ePos,thetaHat)

def genProjections(pulsar,dense):
    Cobeta,Lambda = retrieveCoordinates(pulsar)
    if dense == 0:
        data = getDenseSortedData(pulsar)
    else:
        data = getSortedData(pulsar)
    ssb_pulsar = 1e9*array([sin(Cobeta)*cos(Lambda),sin(Cobeta)*sin(Lambda),cos(Cobeta)])
    thetaVals = []
    phiVals = []
    for i in range(len(data)):
        if dense == 0:
            ssb_earth = array([data[:,1][i],data[:,2][i],data[:,3][i]])*AUperLS
        else:
            ssb_earth = array([data[:,3][i],data[:,4][i],data[:,5][i]])
        
        thetaVals.append(thetaHatProjector(ssb_earth,ssb_pulsar))
        phiVals.append(phiHatProjector(ssb_earth,ssb_pulsar))
    return thetaVals,phiVals    

def window(t,lb,ub):
    if lb < t and ub > t:
        return 1.0
    else:
        return 0.0

def timescaleBoxcar(x,y,s,tau):
    y_smoothed = []
    for i in range(len(x)):
        v1 = 0
        v2 = 0
        tpivot = x[i]
        for j in range(len(x)):
            if abs(tpivot-x[j]) <= tau/2:
                v1 += y[j]/s[j]**2
                v2 += 1.0/s[j]**2
        y_smoothed.append(v1/v2)
    y_smoothed = array(y_smoothed)
    return y_smoothed

###########################
# Compute Basis Functions #
###########################

pulsarTimes = []
pulsarDMs = []
pulsarErrors = []
pulsarGammas = []
for i in range(len(pulsarsToInclude)):
    sortedHolder = getSortedData(pulsarsToInclude[i])
    pulsarTimes.append(sortedHolder[:,0])
    pulsarDMs.append(sortedHolder[:,1])
    pulsarErrors.append(sortedHolder[:,2])

if doIntegrate == 0:
    for i in range(len(pulsarsToInclude)):
        pulsarGammas.append([])
        for j in range(lmax+1):
            for k in range(-j,j+1):
                t,g = genGamma(pulsarsToInclude[i],j,k,1)
                pulsarGammas[i].append(g)
                
else:
    for i in range(len(pulsarsToInclude)):
        pulsarGammas.append([])
        for j in range(lmax+1):
            for k in range(-j,j+1):
                x = genfromtxt('gammaArrays/'+pulsarsToInclude[i]+'_'+str(j)+'_'+str(k)+'.dat')
                if j == 0:
                    pulsarTimes.append(x[:,0])
                pulsarGammas[i].append(x[:,1])

denseTimes = []
denseGammas = []
if doDenseIntegrate == 0:
    for i in range(len(pulsarsToInclude)):
        denseGammas.append([])
        for j in range(lmax+1):
            for k in range(-j,j+1):
                t,g = genGamma(pulsarsToInclude[i],j,k,0)
                denseGammas[i].append(g)
                if j == 0:
                    denseTimes.append(t)

else:
    for i in range(len(pulsarsToInclude)):
        denseGammas.append([])
        for j in range(lmax+1):
            for k in range(-j,j+1):
                x = genfromtxt('gammaArraysDense/'+pulsarsToInclude[i]+'_'+str(j)+'_'+str(k)+'.dat')
                if j == 0:
                    denseTimes.append(x[:,0])
                denseGammas[i].append(x[:,1])
                

####################################
# Apply any masks and do smoothing #
####################################
        
maskedTimes = []
maskedDMs = []
maskedErrors = []
maskedGammas = []
maskedSmoothed = []
for i in range(len(pulsarGammas)):
    masked = 1
    for j in range(len(masks)):
        if masks[j][0] == pulsarsToInclude[i]:
            masked = 0
            lb = masks[j][1]
            ub = masks[j][2]
            maskedTimes.append([])
            maskedDMs.append([])
            maskedErrors.append([])
            maskedGammas.append([])
            Nmoment = 0
            for l in range(lmax+1):
                for m in range(-l,l+1):
                    maskedGammas[i].append([])
                    Nmoment += 1
            for k in range(len(pulsarTimes[i])):
                if pulsarTimes[i][k] < lb or pulsarTimes[i][k] > ub:
                    maskedTimes[i].append(pulsarTimes[i][k])
                    maskedDMs[i].append(pulsarDMs[i][k])
                    maskedErrors[i].append(pulsarErrors[i][k])
                    for l in range(Nmoment):
                        maskedGammas[i][l].append(pulsarGammas[i][l][k])
                

    if masked == 1:
        maskedTimes.append(pulsarTimes[i])
        maskedDMs.append(pulsarDMs[i])
        maskedErrors.append(pulsarErrors[i])
        maskedGammas.append(pulsarGammas[i])

for i in range(len(maskedDMs)):
    maskedSmoothed.append(timescaleBoxcar(maskedTimes[i],maskedDMs[i],maskedErrors[i],tSmooth))
            

maskedThetas = []
maskedPhis = []
denseThetas = []
densePhis = []
for i in range(len(pulsarsToInclude)):
    t,p = genProjections(pulsarsToInclude[i],1)
    tDense,pDense = genProjections(pulsarsToInclude[i],0)
    denseThetas.append(tDense)
    densePhis.append(pDense)
    masked = 1
    for j in range(len(masks)):
        if masks[j][0] == pulsarsToInclude[i]:
            masked = 0
            lb = masks[j][1]
            ub = masks[j][2]
            maskedThetas.append([])
            maskedPhis.append([])
            for k in range(len(pulsarTimes[i])):
                if pulsarTimes[i][k] < lb or pulsarTimes[i][k] > ub:
                    maskedThetas[i].append(t[k])
                    maskedPhis[i].append(p[k])
                
    if masked == 1:
        maskedThetas.append(t)
        maskedPhis.append(p)

###########################
# Collect all coordinates #
###########################
Lambdas = []
Betas = []
for i in range(len(pulsarsToInclude)):
    cb,l = retrieveCoordinates(pulsarsToInclude[i])
    Lambdas.append(l)
    Betas.append(pi/2-cb)


#########################
# Construct Interp Grid #
#########################
tMin = 1e9
tMax = -1e9
for i in range(len(maskedTimes)):
    for j in range(len(maskedTimes[i])):
        if maskedTimes[i][j] < tMin:
            tMin = maskedTimes[i][j]
        if maskedTimes[i][j] > tMax:
            tMax = maskedTimes[i][j]

tauGrid = linspace(tMin,tMax,NGrid)

######################
# Make Design Matrix #
######################

Ncol = 0
Nrow = 0

for i in range(len(maskedTimes)):
    Nrow += len(maskedTimes[i])

for i in range(lmax+1):
    for j in range(-i,i+1):
        Ncol += NGrid
for i in range(len(pulsarsToInclude)):
    tDependent = 1
    Ncol += 1
    for j in range(len(tDependentGradients)):
        if tDependentGradients[j][0] == pulsarsToInclude[i]:
            tDependent = 0
            Ncol += 2*tDependentGradients[j][1]
    if tDependent == 1:
        Ncol += 2

M = zeros([Nrow,Ncol])
row = 0
for i in range(len(pulsarsToInclude)):
    for j in range(len(maskedTimes[i])):
        moment = 0
        col = 0
        for l in range(lmax+1):
            for m in range(-l,l+1):
                time = maskedTimes[i][j]
                gamma = maskedGammas[i][moment][j]
                if NGrid == 1:
                    M[row][col] = gamma
                    col += 1
                else:
                    w = window(time,tauGrid[0],tauGrid[1])
                    M[row][col] = w*gamma*(1.0-(time-tauGrid[0])/(tauGrid[1]-tauGrid[0]))
                    col += 1
                    for k in range(1,NGrid-1):
                        w1 = window(time,tauGrid[k-1],tauGrid[k])
                        w2 = window(time,tauGrid[k],tauGrid[k+1])
                        v1 = (time-tauGrid[k-1])/(tauGrid[k]-tauGrid[k-1])
                        v2 = 1.0-(time-tauGrid[k])/(tauGrid[k+1]-tauGrid[k])
                        M[row][col] = gamma*(w1*v1+w2*v2)
                        col += 1
                    w = window(time,tauGrid[NGrid-2],tauGrid[NGrid-1])
                    M[row][col] = w*gamma*(time-tauGrid[NGrid-2])/(tauGrid[NGrid-1]-tauGrid[NGrid-2])
                    col += 1
                moment += 1
        for k in range(i):
            tDependent = 1
            col += 1
            for l in range(len(tDependentGradients)):
                if pulsarsToInclude[k] == tDependentGradients[l][0]:
                    tDependent = 0
                    col += 2*tDependentGradients[l][1]
            if tDependent == 1:
                col += 2
        thetaHatP = maskedThetas[i][j]
        phiHatP = maskedPhis[i][j]
        tDependent = 1
        for k in range(len(tDependentGradients)):
            if pulsarsToInclude[i] == tDependentGradients[k][0]:
                tDependent = 0
                if tDependentGradients[k][1] == 1:
                    M[row][col] = thetaHatP
                    col += 1
                    M[row][col] = phiHatP
                    col += 1
                else:
                    for l in range(2):
                        if l == 0:
                            proj = thetaHatP
                        elif l == 1:
                            proj = phiHatP
                        subGrid = linspace(min(maskedTimes[i]),max(maskedTimes[i]),tDependentGradients[k][1])
                        w = window(time,subGrid[0],subGrid[1])
                        M[row][col] = w*proj*(1.0-(time-subGrid[0])/(subGrid[1]-subGrid[0]))
                        col += 1
                        for m in range(1,len(subGrid)-1):
                            w1 = window(time,subGrid[m-1],subGrid[m])
                            w2 = window(time,subGrid[m],subGrid[m+1])
                            v1 = (time-subGrid[m-1])/(subGrid[m]-subGrid[m-1])
                            v2 = 1.0-(time-subGrid[m])/(subGrid[m+1]-subGrid[m])
                            M[row][col] = proj*(w1*v1+w2*v2)
                            col += 1
                        w = window(time,subGrid[len(subGrid)-2],subGrid[len(subGrid)-1])
                        M[row][col] = w*proj*(time-subGrid[len(subGrid)-2])/(subGrid[len(subGrid)-1]-subGrid[len(subGrid)-2])
                        col += 1
        if tDependent == 1:
            M[row][col] = thetaHatP
            col += 1
            M[row][col] = phiHatP
            col += 1
        M[row][col] = 1.0
        row += 1



############################
# Make Dense Design Matrix #
############################

NrowDense = 0

for i in range(len(denseTimes)):
    NrowDense += len(denseTimes[i])

Mdense = zeros([NrowDense,Ncol])
row = 0
for i in range(len(pulsarsToInclude)):
    for j in range(len(denseTimes[i])):
        moment = 0
        col = 0
        for l in range(lmax+1):
            for m in range(-l,l+1):
                time = denseTimes[i][j]
                gamma = denseGammas[i][moment][j]
                if NGrid == 1:
                    Mdense[row][col] = gamma
                    col += 1
                else:
                    w = window(time,tauGrid[0],tauGrid[1])
                    Mdense[row][col] = w*gamma*(1.0-(time-tauGrid[0])/(tauGrid[1]-tauGrid[0]))
                    col += 1
                    for k in range(1,NGrid-1):
                        w1 = window(time,tauGrid[k-1],tauGrid[k])
                        w2 = window(time,tauGrid[k],tauGrid[k+1])
                        v1 = (time-tauGrid[k-1])/(tauGrid[k]-tauGrid[k-1])
                        v2 = 1.0-(time-tauGrid[k])/(tauGrid[k+1]-tauGrid[k])
                        Mdense[row][col] = gamma*(w1*v1+w2*v2)
                        col += 1
                    w = window(time,tauGrid[NGrid-2],tauGrid[NGrid-1])
                    Mdense[row][col] = w*gamma*(time-tauGrid[NGrid-2])/(tauGrid[NGrid-1]-tauGrid[NGrid-2])
                    col += 1
                moment += 1
        for k in range(i):
            tDependent = 1
            col += 1
            for l in range(len(tDependentGradients)):
                if pulsarsToInclude[k] == tDependentGradients[l][0]:
                    tDependent = 0
                    col += 2*tDependentGradients[l][1]
            if tDependent == 1:
                col += 2
        thetaHatP = denseThetas[i][j]
        phiHatP = densePhis[i][j]
        tDependent = 1
        for k in range(len(tDependentGradients)):
            if pulsarsToInclude[i] == tDependentGradients[k][0]:
                tDependent = 0
                if tDependentGradients[k][1] == 1:
                    Mdense[row][col] = thetaHatP
                    col += 1
                    Mdense[row][col] = phiHatP
                    col += 1
                else:
                    for l in range(2):
                        if l == 0:
                            proj = thetaHatP
                        elif l == 1:
                            proj = phiHatP
                        subGrid = linspace(min(maskedTimes[i]),max(maskedTimes[i]),tDependentGradients[k][1])
                        w = window(time,subGrid[0],subGrid[1])
                        Mdense[row][col] = w*proj*(1.0-(time-subGrid[0])/(subGrid[1]-subGrid[0]))
                        col += 1
                        for m in range(1,len(subGrid)-1):
                            w1 = window(time,subGrid[m-1],subGrid[m])
                            w2 = window(time,subGrid[m],subGrid[m+1])
                            v1 = (time-subGrid[m-1])/(subGrid[m]-subGrid[m-1])
                            v2 = 1.0-(time-subGrid[m])/(subGrid[m+1]-subGrid[m])
                            Mdense[row][col] = proj*(w1*v1+w2*v2)
                            col += 1
                        w = window(time,subGrid[len(subGrid)-2],subGrid[len(subGrid)-1])
                        Mdense[row][col] = w*proj*(time-subGrid[len(subGrid)-2])/(subGrid[len(subGrid)-1]-subGrid[len(subGrid)-2])
                        col += 1
        if tDependent == 1:
            Mdense[row][col] = thetaHatP
            col += 1
            Mdense[row][col] = phiHatP
            col += 1
        Mdense[row][col] = 1.0
        row += 1



##############
# Do fitting #
##############

stackedHipass = []
for i in range(len(maskedDMs)):
    for j in range(len(maskedDMs[i])):
        stackedHipass.append(maskedDMs[i][j]-maskedSmoothed[i][j])

C = zeros([len(stackedHipass),len(stackedHipass)])
Cinv = zeros([len(stackedHipass),len(stackedHipass)])
index = 0
for i in range(len(maskedErrors)):
    for j in range(len(maskedErrors[i])):
        C[index][index] = maskedErrors[i][j]**2
        Cinv[index][index] = maskedErrors[i][j]**-2
        index += 1

CpInv = dot(transpose(M),dot(Cinv,M))
Cp = linalg.inv(CpInv)

deltaP = dot(Cp,dot(transpose(M),dot(Cinv,stackedHipass)))


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
            val += deltaP[k]*M[j+shift][k]
        models[i].append(val)
    shift += len(maskedTimes[i])
    for j in range(len(denseTimes[i])):
        val = 0
        sunval = 0
        pxval = 0
        momentVals = zeros(Nmoment)
        for k in range(len(deltaP)):
            val += deltaP[k]*Mdense[j+shiftDense][k]
            if k < NGrid*Nmoment:
                sunval += deltaP[k]*Mdense[j+shiftDense][k]
                momentVals[k/NGrid] += deltaP[k]*Mdense[j+shiftDense][k]
            else:
                pxval += deltaP[k]*Mdense[j+shiftDense][k]
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
        residuals[i].append(maskedDMs[i][j]-maskedSmoothed[i][j]-models[i][j])


################
# Compute a X2 #
################

X2 = 0
nDOF = -len(deltaP)
for i in range(len(residuals)):
    for j in range(len(residuals[i])):
        X2 += (residuals[i][j]/maskedErrors[i][j])**2
        nDOF += 1

print X2,nDOF,X2/nDOF

##############
# Make plots #
##############


def makePlot(index):
    pulsar = pulsarsToInclude[index]
    beta = Betas[index]

    fig = figure(figsize=(15,12))
    #ax1 = fig.add_axes([.15,.79,.8,.15])
    ax2 = fig.add_axes([.1,.27,.85,.70])
    ax3 = fig.add_axes([.1,.10,.85,.15])
    #ax1.set_xticks([])
    ax2.set_xticks([])

    ax2.set_title(pulsar+'  '+r'$(\beta=$'+str(beta*180/pi)[0:5]+r'$^\circ)$',size=18)

    
    ax2.set_ylabel(r'${\rm DMX}$'+' '+r'$[{\rm pc}$'+' '+r'${\rm cm}^{-3}]$',size=18)
    ax3.set_ylabel(r'${\rm Resids.}$',size=18)
    ax3.set_xlabel(r'${\rm Time}$'+' '+r'$[{\rm MJD}]$',size=18)

    #ax1.plot(denseTimes[index],modelsDense[index],'r')
    #ax1.plot(denseTimes[index],pxModelsDense[index],'g')
    #ax1.plot(denseTimes[index],sunModelsDense[index],'m')
    
    ax2.plot(maskedTimes[index],maskedSmoothed[index],'b')
    ax2.errorbar(pulsarTimes[index],pulsarDMs[index],pulsarErrors[index],fmt='k.')
    ax2.plot(maskedTimes[index],models[index]+maskedSmoothed[index],'r')
    ax2.plot(maskedTimes[index],models[index]+maskedSmoothed[index],'r.',markersize=8)

    ax3.errorbar(maskedTimes[index],residuals[index],maskedErrors[index],fmt='k.')
    ax3.plot([tMin,tMax],[0,0],'k')

    #ax1.set_xlim([tMin,tMax])
    ax2.set_xlim([tMin,tMax])
    ax3.set_xlim([tMin,tMax])

    fig.savefig('officialDMPlots/'+pulsar+'.png')
    close()


if doPlot == 0:
    for i in range(len(pulsarsToInclude)):
        makePlot(i)
