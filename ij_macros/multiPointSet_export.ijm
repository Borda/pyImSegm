/*
 * @file export_MultipointSet.ijm
 * @title Marco for exporting multi-point set
 * @author Jiri Borovec
 * @date 13/06/2014
 * @mail jiri.borovec@fel.cvut.cz
 * 
 * @brief: This macro does export set of points from Multi-point tool 
 * into .csv and .txt files (the name is specified during exporting)
 */

// clean results
run("Clear Results");
// get all points
getSelectionCoordinates(xCoordinates, yCoordinates);

// chose name pattern for exporting
//fileName = File.openDialog("Select the file for export");
// just to have name of file...
f = File.open("");
File.close(f)
// remove this temporar file
res = File.delete(File.directory + File.name)

tmp = split(File.name,".");
fileName = File.directory + tmp[0];

// Exporting as TXT format (ITK compatible)
file = File.open(fileName+".txt");
print(file, "point");
print(file, lengthOf(xCoordinates) );
for(i=0; i<lengthOf(xCoordinates); i++) {
    print(file, xCoordinates[i] + " " + yCoordinates[i]);
}
File.close(file)

// export as CSV file
for(i=0; i<lengthOf(xCoordinates); i++) {
    setResult("X", i, xCoordinates[i]);
    setResult("Y", i, yCoordinates[i]);
}
updateResults();
saveAs("Results", fileName+".csv"); 
