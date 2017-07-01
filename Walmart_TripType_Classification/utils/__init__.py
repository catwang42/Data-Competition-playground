import numpy as np
from sklearn import preprocessing

def train_valid_split(trainfile):
    # Now split the train data into train+valid.
    #  2.5% with seed 0 appears to be pretty good, at least until this gets competitive
    visits = trainfile["VisitNumber"].unique()

    # determine visits to go in validation set
    np.random.seed(0)
    validation_visits = np.random.choice(visits, int(len(visits) * .025))
    validation_set = (trainfile["VisitNumber"] == validation_visits[0])

    for i in range(1, len(validation_visits)):
        validation_set |= trainfile["VisitNumber"] == validation_visits[i]
    valid = trainfile.ix[validation_set]

    # flip that around to get the train set
    training_set = np.invert(validation_set)
    train = trainfile.ix[training_set]
    return  (train,valid)


def make_ddcomb(df,train, num_visits=100000000):
    # make categorical #'s.
    ddcat = np.unique(train["DepartmentDescription"])

    # build map
    ddmap = {}
    for i in range(len(ddcat)):
        ddmap[ ddcat[i] ] = i

    # Construct set of usable DepartmentDescription keys (with >1000 per dept)
    train_psc = train[train.ScanCount >= 1].DepartmentDescription
    vc = train_psc.value_counts()

    ddi_len = np.zeros(len(ddcat))
    ddi_keys = {}
    for i in vc.iteritems():
        ddi_len[i[0]] = i[1]
        if i[1] > 1000:
            ddi_keys[i[0]] = True

    flmap = {}
    ddmax = np.max(train["DepartmentDescription"]) + 1

    # # 6-ddmax: coarse dept description
    fnum = 6 + ddmax

    # # rest of fnum: fineline mapping
    for cat in range(0, ddmax):
        # for cat in [20]:
        catmask = train[train.ScanCount >= 1,train.DepartmentDescription == cat]
        subset = train_psc[catmask]
        if len(subset) < 10:
            continue

        vc = subset.FinelineNumber.value_counts()

        for iterit in vc.iteritems():
            fln = iterit[0]
            if (iterit[1] < 10):  # or (len(subset) < (iterit[1] * 2)):
                continue
            fnum += 1
            flmap[(cat, fln)] = fnum

    num_ents = len(df)
    visits = np.sort(df["VisitNumber"].unique())
    num_visits = min(num_visits, len(visits))
    ddmax = np.max(train["DepartmentDescription"])

    mat = np.zeros((num_visits, (ddmax * 1) + fnum + 2))
    tt = np.zeros(num_visits)
    df_scancount = df.ScanCount.values
    df_visitnumber = df.VisitNumber.values
    df_triptype = df.TripType.values if ('TripType' in df) else np.zeros(len(df))
    df_ddint = df.DepartmentDescription.values
    df_fln = df.FinelineNumber.values
    visitmap = {}
    vnum = -1
    icount = np.zeros(num_visits + 1)

    for i in range(0, num_ents):
        try:
            visit = visitmap[df_visitnumber[i]]
        except:
            vnum += 1
            if (vnum + 1) == num_visits:
                break
            visitmap[df_visitnumber[i]] = vnum
            visit = vnum
            tt[vnum] = df_triptype[i]
        icount[visit] += 1
        if True:  # df_scancount[i] > 0:
            dept = df_ddint[i]
            fln = df_fln[i]
            mat[visit][6 + df_ddint[i]] += (1 + ((df_scancount[i] - 1) * .25))
            try:
                feature = flmap[(dept, fln)]
                mat[visit][feature] += (1 + ((df_scancount[i] - 1) * .25))
            except:
                None

        if df_scancount[i] < 0:
            mat[visit][0] += 1
    vnum += 1
    f_ddstart = 6
    f_ddend = 6 + ddmax
    for i in range(0, vnum):
        if np.sum(mat[i][f_ddstart:f_ddend]):
            mat[i][f_ddstart:f_ddend] /= np.sum(mat[i][f_ddstart:f_ddend])
        if np.sum(mat[i][f_ddend:fnum]):
            mat[i][f_ddend:fnum] /= np.sum(mat[i][f_ddend:fnum])
        if icount[i] > 0:
            mat[i][0] /= icount[i]
            if icount[i] == 1:
                mat[i][1] = 1
            elif icount[i] == 2:
                mat[i][2] = 1
            elif icount[i] < 5:
                mat[i][3] = 1
            elif icount[i] < 10:
                mat[i][4] = 1
            else:
                mat[i][5] = 1
    return mat, visits, tt