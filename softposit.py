import numpy as np


# softposit for pose estimation, a translation of softposit.m by Danial DeMenthon and Philip David
# http://www.daniel.umiacs.io/SoftPOSIT.txt
# Licensed under GPL

def maxPosRatio(assignMat):
    pos = []
    ratios = []

    nrows, ncols = assignMat.shape

    nimgpts = nrows
    nmodpnts = ncols

    for k in range(nmodpnts - 1):
        vmax, imax = np.max(assignMat[:, k]), np.argmax(assignMat[:, k])

        if imax + 1 == nrows:
            continue
        # check if the max value in the column is also max within its row
        if vmax >= np.max(assignMat[imax, :]):
            pos.append([imax, k])

            rr = assignMat[imax, ncols - 1] / assignMat[imax, k]
            cr = assignMat[nrows - 1, k] / assignMat[imax, k]

            ratios.append([rr, cr])

    return pos, ratios


def proj3dto2d(pts3d, rot, trans, focal, center):
    original_shape = pts3d.shape
    if pts3d.shape[0] != 3:
        pts3d = pts3d.T

    numpts = pts3d.shape[1]

    campts = rot @ pts3d + trans[:, np.newaxis]
    campts[2, :] = np.maximum(campts[2, :], 1e-10)

    pts2d = focal + campts[0:2, :] / campts[2, :]
    pts2d = pts2d + center[:, np.newaxis]

    if original_shape[0] != 3:
        pts2d = pts2d.T
    return pts2d


def sinkhornSlack(M):
    iMaxIterSinkhorn = 60
    fEpsilon2 = 0.001

    iNumSinkIter = 0
    nbRows, nbCols = M.shape

    fMdiffSum = fEpsilon2 + 1

    while (abs(fMdiffSum) > fEpsilon2 and iNumSinkIter < iMaxIterSinkhorn):
        Mprev = M
        # col normalization
        McolSums = np.sum(M, axis=0)
        McolSums[nbCols - 1] = 1  # dont normalize slack col terms against each other
        McolSumsRep = np.ones((nbRows, 1)) * McolSums
        M = M / McolSumsRep
        # row normalization
        MrowSums = np.sum(M, axis=1)
        MrowSums[nbRows - 1] = 1
        MrowSumsRep = MrowSums * np.ones((1, nbCols))
        M = M / MrowSumsRep

        iNumSinkIter += 1
        fMdiffSum = np.sum(np.abs(M[:] - Mprev[:]))

    return M


def sinkhornImp(M):
    iMaxIterSinkhorn = 60
    fEpsilon2 = 0.001

    iNumSinkIter = 0
    nbRows, nbCols = M.shape

    fMdiffSum = fEpsilon2 + 1
    posmax, ratios = maxPosRatio(M)

    while abs(fMdiffSum) > fEpsilon2 and iNumSinkIter < iMaxIterSinkhorn:
        Mprev = M
        # col normalization
        McolSums = np.sum(M, axis=0)
        McolSums[nbCols - 1] = 1
        McolSumsRep = np.ones((nbRows, 1)) * McolSums
        M = M / McolSumsRep

        # fix vals in slack column
        for i in range(len(posmax)):
            M[posmax[i][0], nbCols - 1] = ratios[i][0] * M[posmax[i][0], posmax[i][1]]

        # row normalization, except for slack row
        MrowSums = np.sum(M, axis=1)
        MrowSums[nbRows - 1] = 1
        MrowSumsRep = MrowSums[np.newaxis].T @ np.ones((1, nbCols))
        M = M / MrowSumsRep

        # fix vals in slack row
        for i in range(len(posmax)):
            M[nbRows - 1, posmax[i][1]] = ratios[i][1] * M[posmax[i][0], posmax[i][1]]

        iNumSinkIter += 1
        fMdiffSum = np.sum(np.abs(M[:] - Mprev[:]))

    return M


def nummatches(assignMat):
    num = 0;
    nrows, ncols = assignMat.shape
    nimgpnts, nmodpts = nrows, ncols

    for k in range(nmodpts):
        vmax, imax = np.max(assignMat[:, k]), np.argmax(assignMat[:, k])
        if imax == nrows - 1:
            continue
        if vmax >= np.max(assignMat[imax, :]):
            num += 1

    return num


def softposit(imagepts, imageadj, worldpts, worldadj, beta0, noisestd, initrot, inittrans, focal, displevel, kickout,
              center):
    stats = []
    alpha = 9.21 * noisestd ** 2 + 1
    maxDelta = np.sqrt(alpha) / 2
    betaFinal = 0.5
    betaUpdate = 1.05
    epsilon0 = 0.01

    maxCount = 2
    minBetaCount = 20

    nbImagePts = imagepts.shape[0]
    nbWorldPts = worldpts.shape[0]

    minNbPts = min(nbImagePts, nbWorldPts)
    maxNbPts = max(nbImagePts, nbWorldPts)

    scale = 1 / (maxNbPts + 1)

    centeredImage = np.zeros_like(imagepts)
    centeredImage[:, 0] = (imagepts[:, 0] - center[0]) / focal
    centeredImage[:, 1] = (imagepts[:, 1] - center[1]) / focal

    imageOnes = np.ones((nbImagePts, 1))

    worldOnes = np.ones((nbWorldPts, 1))
    homogenousWorldPts = np.block([worldpts, worldOnes])
    # homogenousWorldPts = np.asarray(homogenousWorldPts)

    rot = initrot
    trans = inittrans

    if displevel in [5, 6]:
        print('Initial transforms are:', initrot, inittrans)

    # init depts of world points based on initial transform
    wk = homogenousWorldPts @ np.block([rot[2, :] / trans[2], 1])[np.newaxis].T  # ?

    # first two rows of camera matrix

    r1T = np.block([rot[0, :] / trans[2], trans[0] / trans[2]]).T
    r2T = np.block([rot[1, :] / trans[2], trans[1] / trans[2]]).T  # ?

    betaCount = 0
    poseConverged = False
    assignConverged = False
    foundPose = False
    beta = beta0

    # the assignment matrix also has a slack row and column

    assignMat = np.ones((nbImagePts + 1, nbWorldPts + 1)) + epsilon0

    while beta < betaFinal and not assignConverged:
        projectedU = homogenousWorldPts @ r1T
        projectedV = homogenousWorldPts @ r2T

        replicatedProjectedU = imageOnes @ projectedU[np.newaxis]
        replicatedProjectedV = imageOnes @ projectedV[np.newaxis]

        wkxj = centeredImage[:, 0][np.newaxis].T @ wk.T
        wkyj = centeredImage[:, 1][np.newaxis].T @ wk.T

        # exit(0)

        distMat = (focal * focal) * ((replicatedProjectedU - wkxj) ** 2 + (replicatedProjectedV - wkyj) ** 2)

        if displevel in [5, 6]:
            print('distMat is:', distMat)

        # use softassign to compute best assignment with computed distances

        assignMat[0:nbImagePts, 0:nbWorldPts] = scale * np.exp(-beta * (distMat - alpha))
        assignMat[0:nbImagePts + 1, nbWorldPts] = scale
        assignMat[nbImagePts, 0:nbWorldPts + 1] = scale

        assignMat = sinkhornImp(assignMat)

        if displevel in [5, 6]:
            print('Processed assignMat is:', assignMat)

        numMatchPts = nummatches(assignMat)
        sumNonslack = np.sum(assignMat[0:nbImagePts, 0:nbWorldPts])

        # use posit to caclulate new pose

        summedByColAssign = np.sum(assignMat[0:nbImagePts, 0:nbWorldPts], axis=0)

        sumSkSkT = np.zeros((4, 4))

        for k in range(nbWorldPts):
            sumSkSkT += summedByColAssign[k] * homogenousWorldPts[k, :][np.newaxis].T @ homogenousWorldPts[k, :][
                np.newaxis]

        if np.linalg.cond(sumSkSkT) > 1e10:
            print('error: sumSkSkT is ill-conditioned')
            return

        objectMat = np.linalg.inv(sumSkSkT)

        poseConverged = 0;
        count = 0;

        r1Tprev = r1T;
        r2Tprev = r2T;

        # posit loop. we do a single round so assignments and pose converge together

        weightedUi = np.zeros((4, 1))
        weightedVi = np.zeros((4, 1))

        for j in range(nbImagePts):
            for k in range(nbWorldPts):
                weightedUi += assignMat[j, k] * wk[k] * centeredImage[j, 0] * homogenousWorldPts[k, :][np.newaxis].T
                weightedVi += assignMat[j, k] * wk[k] * centeredImage[j, 1] * homogenousWorldPts[k, :][np.newaxis].T

        r1T = objectMat @ weightedUi
        r2T = objectMat @ weightedVi

        use_chang_and_tsai = True

        if use_chang_and_tsai:
            svd_target = np.block([[r1T[0:3].T], [r2T[0:3].T]]).T
            U, S, V = np.linalg.svd(svd_target)  # ?
            A = U @ np.array([[1, 0], [0, 1], [0, 0]]) @ V  # Not the same V in numpy as matlab
            r1 = A[:, 0]
            r2 = A[:, 1]
            r3 = np.cross(r1, r2)
            Tz = 2 / (S[0] + S[1])
            Tx = r1T[3] * Tz
            Ty = r2T[3] * Tz
            r3T = np.block([r3, Tz])
        else:
            raise NotImplementedError

        r1T = np.block([r1, Tx]) / Tz
        r2T = np.block([r2, Ty]) / Tz

        wk = homogenousWorldPts @ r3T / Tz
        wk = wk[:, np.newaxis]

        delta = np.sqrt(np.sum(assignMat[0:nbImagePts, 0:nbWorldPts] * distMat) / nbWorldPts)

        poseConverged = delta < maxDelta

        if displevel in [3, 4, 5, 6]:
            print('betaCount is:', betaCount)
            print('beta is:', beta)
            print('delta is:', delta)
            print('poseConverged is:', poseConverged)
            print('numMatchPts is:', numMatchPts)
            print('sumNonslack is:', sumNonslack)

        stats.append([beta, delta, numMatchPts / nbWorldPts, sumNonslack / nbWorldPts,
                      np.sum(np.array([(r1T - r1Tprev).T, (r2T - r2Tprev).T]) ** 2)])

        count += 1

        # end of posit loop

        beta = beta * betaUpdate
        betaCount += 1
        assignConverged = poseConverged and betaCount > minBetaCount

        trans = np.block([Tx, Ty, Tz])
        rot = np.block([r1[np.newaxis].T, r2[np.newaxis].T, r3[np.newaxis].T]).T

        if displevel in [5, 6]:
            print('Current transform is ', rot, trans)

        foundPose = delta < maxDelta and betaCount > minBetaCount

        projWorldPts = proj3dto2d(worldpts, rot, trans, focal, center)

        r = numMatchPts / kickout.numMatchable
        if r < kickout.rthldfbeta[betaCount]:
            if displevel in [3, 4, 5, 6]:
                print('Terminating due to low match ratio, early restart', r, betaCount, kickout.rthldfbeta(betaCount))
            return

    if poseConverged:
        print("Converged!")

        initprojpts = proj3dto2d(worldpts, initrot, inittrans, focal, center)
        finalprojpts = proj3dto2d(worldpts, rot, trans, focal, center)
        print("Final transform: ", rot, trans, sep='\n')

    return rot, trans, assignMat, projWorldPts, foundPose, stats


class kickout:
    def __init__(self, matchable=5, rthldfbeta=np.zeros((200, 1))):
        self.numMatchable = matchable
        self.rthldfbeta = rthldfbeta