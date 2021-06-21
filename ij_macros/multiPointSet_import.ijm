/*
 * @file import_MultipointSet.ijm
 * @title Marco for exporting multi-point set
 * @author Jiri Borovec
 * @date 13/06/2014
 * @mail jiri.borovec@fel.cvut.cz
 *
 * @brief: This macro does importing set of points from Multi-point tool
 * from .csv and .txt files (the name is specified during exporting)
 */

// ask for a file to be imported
fileName = File.openDialog("Select the file to import");
allText = File.openAsString(fileName);
tmp = split(fileName,".");
// get file format {txt, csv}
ext = tmp[lengthOf(tmp)-1];
// parse text by lines
text = split(allText, "\n");

// define array for points
var xpoints = newArray;
var ypoints = newArray;

// in case input is in TXT format
if (ext=="txt") {
	print("importing TXT point set...");
	//these are the column indexes
	//hdr = split(text[0]);
	nbPoints = split(text[1]);
	iX = 0; iY = 1;
	// loading and parsing each line
	for (i = 2; i < (text.length); i++){
	   line = split(text[i]," ");
	   setOption("ExpandableArrays", true);
	   xpoints[i-2] = parseInt(line[iX]);
	   ypoints[i-2] = parseInt(line[iY]);
	   print("p("+i-1+") ["+xpoints[i-2]+"; "+ypoints[i-2]+"]");
	}
// in case input is in CSV format
} else if (ext=="csv") {
	print("importing CSV point set...");
	//these are the column indexes
	//hdr = split(text[0]);
	if (text[0]==',X,Y' || text[0]==' ,X,Y') {
	   iLabel = 0; iX = 1; iY = 2;
	} else {
	   iX = 0; iY = 1;
	}
	// loading and parsing each line
	for (i = 1; i < (text.length); i++){
	   line = split(text[i],",");
	   setOption("ExpandableArrays", true);
	   xpoints[i-1] = parseInt(line[iX]);
	   ypoints[i-1] = parseInt(line[iY]);
	   print("p("+i+") ["+xpoints[i-1]+"; "+ypoints[i-1]+"]");
	}
// in case of any other format
} else {
	print("not supported format...");
}

// show the points in the image
makeSelection("point", xpoints, ypoints);
