/*
Convert all images in input folder from tif to png format
and perform colour adjustment so the objest are nice to see..
(eg. Drosophila 2d slices)

run in commad line
 >> Fiji.app/ImageJ-linux64 -batch convert_tif2png.ijm

Copyright (C) 2016 Jiri Borovec <jiri.borovec@fel.cvut.cz>
*/

dirIn="/mnt/F464B42264B3E590/DATA/Drosophila/ovary_2d/slices/"
dirOut="/mnt/F464B42264B3E590/DATA/Drosophila/ovary_2d/slices_gray/"

//dirIn="/mnt/F464B42264B3E590/DATA/Drosophila/ovary_2d/stage5/"
//dirOut="/mnt/F464B42264B3E590/DATA/Drosophila/ovary_2d/stage5_gray/"

// Display info about the files
list = getFileList(dirIn);
print("found: " + toString(list.length));

close("All")

for (i=0; i<list.length; i++){
	name = list[i];
	nameNew = replace(name, ".tif", ".png");
	
	print(toString(i+1) + "#" + toString(list.length) + " -> " + name);	
	open(dirIn + name);
	
	//run("Brightness/Contrast...");
	run("Stack to Images");
	selectWindow("Green");
	close();
	selectWindow("Blue");
	close();
	selectWindow("Red");
	run("8-bit");
	run("8-bit");
	
	saveAs("PNG", dirOut + nameNew);	
	close();
}
