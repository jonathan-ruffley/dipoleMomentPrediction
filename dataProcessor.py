#there are two csv files. One has dipole moments for ~205 molecules, and the other has dipole moments for ~470 molecules, including everything in the first file. The first file already has smiles strings that can be processed with rdkit the second file does not. Here I need to create a new csv that only has the molecules that do not yet have a smiles string
import pandas as pd

#load the organic only dataset into dataframe
organicDF = pd.read_csv('./organicMolecules.csv')
#print(organicDF.head())
#convert the Molecule column to a list
organicMolecules = organicDF['Molecule'].tolist()
#print(organicMolecules)
#now have a list of all the organic molecules in the original dataset. Now, load the full dataset, make a similar list of the Molecules column, and drop a column if it matches what is in the organic dataset
fullDF = pd.read_csv('./allDipoleMoments.csv')
#print(fullDF.head())
allMolecules = fullDF['Molecule'].tolist()
#print(allMolecules)

#now have two lists. For each molecule in full data set, check if it is in the organic dataset. If it is, drop it from the full data frame
for item in allMolecules:
    if item in organicMolecules:
        #print(item)
        #drop the item from full dataframe
        indexNames = fullDF[ fullDF['Molecule'] == item].index
        fullDF.drop(indexNames, inplace=True)

#test if this worked by printing
print(fullDF)
#it works! print to another file that can be combined with the DipoleMoments.csv later
fullDF.to_csv('./inorganicMolecules.csv')
