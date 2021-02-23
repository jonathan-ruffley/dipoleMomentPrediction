#run with miniconda (necessary for rdkit).
#code will take a csv of dipole moments and build a random forest algorithm to predict dipole moments
import pandas as pd
import rdkit.Chem as chem
from rdkit.Chem import AllChem
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import statistics as s
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
import scipy as sci

#options
pd.set_option('display.max_rows', 500)

#read the data files, verify types, drop the structure column as it will not be used in the analysis here.
df = pd.read_csv('./organicMolecules.csv')

#read the inorganic and high dipole moment data
dfInorganic = pd.read_csv('./inorganicMolecules.csv')
dfHigh = pd.read_csv('./highDipoleMoment.csv')
dfOther = pd.read_csv('./otherMolecules.csv')

#combine these extra data frames. They will be only used for training, not testing. There isn't enough data on inorganic molecules to make good predictions of inorganics, and there isn't enough data with dipole moments greater than 4 to make good predictions in that domain.
dfSupplement = pd.concat([dfInorganic, dfHigh, dfOther], ignore_index=True)
dfSupplement.drop(['Structure','Molecule'], axis=1, inplace=True)
df.drop('Structure', axis=1, inplace=True)

#convert smiles strings into 1024 bit morgan (circular) fingerprint
#first convert smiles string column into numpy array
smilesStringArray = df['smiles string'].to_numpy('object')
#repeat for supplemental dataset
supplementalSmiles = dfSupplement['smiles string'].to_numpy('object')

#convert into array of fingerprints
fingerprintList= []
for smile in smilesStringArray:
    mol = chem.MolFromSmiles(smile)
    fingerprint = chem.AllChem.GetMorganFingerprintAsBitVect(mol, 2, useChirality=True, useBondTypes=True, nBits=1024)
    fingerprint = np.array(list(fingerprint.ToBitString()))
    #print('{0}'.format(fingerprint))
    fingerprintList.append(fingerprint)
fingerprintArray = np.array(fingerprintList)
fingerprintArray = fingerprintArray.astype(np.float).astype(int).astype(str)

#repeat for supplemental data
supplementalFingerprintList= []
for smile in supplementalSmiles:
    mol = chem.MolFromSmiles(smile)
    fingerprint = chem.AllChem.GetMorganFingerprintAsBitVect(mol, 2, useChirality=True, useBondTypes=True, nBits=1024)
    fingerprint = np.array(list(fingerprint.ToBitString()))
    #print('{0}'.format(fingerprint))
    supplementalFingerprintList.append(fingerprint)
supplementalFingerprintArray = np.array(supplementalFingerprintList)
supplementalFingerprintArray = supplementalFingerprintArray.astype(np.float).astype(int).astype(str)
#print(supplementalFingerprintArray)

#convert back to part of the data frame, adding 1024 columns, then combine the dataframes, in the order morgan fingerprints, dipole moment. This removes the smiles string column; it is no longer needed
dfFingerprints = pd.DataFrame(fingerprintArray)
supplementalFingerprintDF = pd.DataFrame(supplementalFingerprintArray)
finalDF = pd.concat([dfFingerprints, df['DipoleMoment']], axis=1)
finalSupplement = pd.concat([supplementalFingerprintDF, dfSupplement['DipoleMoment']], axis=1)
fullDataSet = pd.concat([finalDF, finalSupplement], axis=0, ignore_index = True)
#print(fullDataSet.tail(300))

#summary statistics on the dipole moments
dipoleMoments = df['DipoleMoment'].to_numpy('object')
mean = s.mean(dipoleMoments)
stddev = s.stdev(dipoleMoments)
skewness = sci.stats.skew(dipoleMoments)
kurtosis = sci.stats.kurtosis(dipoleMoments)
print('Mean: {0:.3f} D'.format(mean))
print('Standard Deviation: {0:.3f} D'.format(stddev))
print('Skew: {0:.3f}'.format(skewness))
print('Kurtosis {0:.3f}'.format(kurtosis))
sb.distplot(df['DipoleMoment'], 16, kde=False)
plt.xlabel('Dipole Moment (D)')
plt.ylabel('Instances')
plt.savefig('DipoleMomentHistogram.png')
plt.show()


#the data frame is ready, now it's time for the random forest.
#split data into train and test
model = RandomForestRegressor(n_estimators=1600, min_samples_split=5, min_samples_leaf=2, max_features='auto', max_depth=None, bootstrap=True)
folds = KFold(n_splits=5, shuffle=True)

scores = []
mae = []
mse = []
mape = []
data = finalDF.drop(['DipoleMoment'], axis=1).values
target = finalDF.DipoleMoment.values
fullData = fullDataSet.drop(['DipoleMoment'], axis=1)
fullTarget = fullDataSet.DipoleMoment.values

x = np.linspace(0, 4.1)
y = x
print('\nScore:\n')
for trainIndex, testIndex in folds.split(data):
    plt.figure()
    xTrain = np.concatenate((data[trainIndex],finalSupplement.drop('DipoleMoment',axis=1).values))
    xTest = data[testIndex]
    yTrain = np.concatenate((target[trainIndex],finalSupplement.DipoleMoment.values))
    yTest = target[testIndex]
    #print('\n\n{0}\n\n{1}\n\n{2}\n\n{3}'.format(xTrain,xTest,yTrain,yTest))
    fittage = model.fit(xTrain, yTrain)
    SCORE = model.score(xTest, yTest) 
    print('{0:.3f}'.format(SCORE))
    yHat = model.predict(xTest)
    mae.append(metrics.mean_absolute_error(yTest, yHat))
    mse.append(metrics.mean_squared_error(yTest, yHat))
    #mape.append(metrics.mean_absolute_percentage_error(yTest, yHat))
    #RESIDUAL PLOTS
    #plt.scatter(yTest, yHat-yTest)
    #plt.plot([0,4],[0,0], 'k--')
    #residualPlot = sb.residplot(yTest, yHat)
    #plt.xlabel('Dipole Moment (D)')
    #plt.ylabel('Error (D)')
    #plt.show()
    #PARITY PLOTS
    plt.scatter(yTest, yHat)
    plt.plot(x, y, 'k--')
    plt.xlabel('Dipole Moment (D)')
    plt.ylabel('Predicted Dipole Moment (D)')
    plt.axis([-0.1,4,-0.1,4])
    plt.draw()
    scores.append(SCORE)
plt.show()
meanScore = s.mean(scores)
print('Model score is: {0:.3f}\n'.format(meanScore))
print('MAE: {0:.3f} D'.format(s.mean(mae)))
print('MSE: {0:.3f}'.format(s.mean(mse)))
#print('MAPE: {0:.3f}'.format(s.mean(mape)))


#optimize the hyperparameters. The most important hyperparameters are the number of estimators, the maximum number of features per node, the max depth of each tree, minimum date points in a node before splitting the node, minimum number of data points allowed in a leaf node, bootstrap


#hyperparameter grid
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# max_features = ['auto', 'sqrt']
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None) #determine what this line does and if it is necessary
# min_samples_split = [2, 5, 10]
# min_samples_leaf = [1, 2, 4]
# bootstrap = [True, False]

# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}

# modelRandom = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 200, cv = 5, verbose = 2, random_state = 22, n_jobs = -1)
# modelRandom.fit(xTrain, yTrain)

# print(modelRandom.best_params_)
#the best parameters are 1600 estimators, 5 samples split, min samples leaf 2, auto max features, max depth None, bootstrap True.

#now that the train test is known, train on the the entire dataset so that it can be used for predictions
modelSettings = model.fit(fullData, fullTarget)
#print(modelSettings)
#Save the model with pickle or json? Add deployment later
#filename = 'dipoleMomentModel.sav'

#allow input of new predictions
prediction = True
print('\n\nEnter a smiles string for a new molecule.\nInorganic or large molecules (over 10 atoms) may not result in accurate predictions.\nThere are some limitations on what the interpreter can handle,\nso you may get a rejection for a valid smile.\nSee the RDkit documentation for more details.\n')
while prediction == True:
    repeat = True
    newSmile = input('\nEnter a smiles string for a new molecule.\n')
    try:
        #convert smile to MF
        mol = chem.MolFromSmiles(newSmile)
        fingerprint = chem.AllChem.GetMorganFingerprintAsBitVect(mol, 2, useChirality=True, useBondTypes=True, nBits=1024)
        fingerprint = np.array(list(fingerprint.ToBitString()))
        #calculate a dipole moment for smile
        newSmileDipole = model.predict(fingerprint.reshape(1,-1))
        print('\nPredicted dipole moment for {0} is: {1:.3f} D\n'.format(newSmile, float(newSmileDipole[0])))
    except:
        print('Smile input error, try again.\n')
        repeat = False
    while repeat == True:
        moreSmiles = input('\nDo you need another dipole moment? (y/n)\n')
        if moreSmiles == 'y' or moreSmiles == 'n':
            if moreSmiles == 'y':
                repeat = False
            elif moreSmiles == 'n':
                repeat = False
                prediction = False
            else:
                print('error in prediction loop')
        else:
            print('\nIncorrect entry, try again.\n')
        
   # except:
    #    print('\nSmile string may be invalid, try again.\n')
