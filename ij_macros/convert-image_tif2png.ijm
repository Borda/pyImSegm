/*
Convert all images in input folder from tif to png format
and perform colour adjustment so the objest are nice to see..
(eg. Drosophila 2d slices)

run in commad line
 >> ~/Applications/Fiji.app/ImageJ-linux64 -batch convert_tif2png.ijm

Copyright (C) 2016 Jiri Borovec <jiri.borovec@fel.cvut.cz>
*/

dirIn="/datagrid/Medical/microscopy/drosophila/ovary_selected_images/"
dirOut="/datagrid/Medical/microscopy/drosophila/RESULTS/PIPELINE_ovary_selected_images/0_input_images_png/"

//dirIn="/home/jirka/Dropbox/Workspace/segment_Medical/Drosophila-Ovary/ovary_2d/stages/"
//dirOut="/home/jirka/Dropbox/Workspace/segment_Medical/Drosophila-Ovary/ovary_2d/stages_png/"

// Display info about the files
list = getFileList(dirIn);
print("found: " + toString(list.length));

for (i=0; i<list.length; i++){
	name = list[i];
	nameNew = replace(name, ".tif", ".png");

	print(toString(i+1) + "#" + toString(list.length) + " -> " + name);
	open(dirIn + name);

	//run("Brightness/Contrast...");
	run("Enhance Contrast", "saturated=0.35");

	saveAs("PNG", dirOut + nameNew);
	close();
}
