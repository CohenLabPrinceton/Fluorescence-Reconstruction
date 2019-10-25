/* 
This FIJI macro requires the user to define the following inputs:
n - (integer) the number of patches to split the input images into. 
	For example, if your input images are of size 512x512, n = 2 to obtain 4 patches of 256x256 each. 
in_dir - (string) the path to your input files. 
		These height and width of these images should be divisible by 256. Use the makeRectangle() command if needed. 
out_dir - (string) the path to a folder into which output patch images will be saved. 
var_name - (string) the name of the variable

The macro will save the newly created image patches into the folder defined in out_dir. 
*/ 

// -----------------------------------------------------
// User-defined parameters:
n = 2; 
in_dir = getDirectory("home") + "Sample_Images/Phase_Image/"
out_dir = getDirectory("home") + "Sample_Images/PHASE/"
var_name = "phase"
// -----------------------------------------------------

// Get all the files in the input directory and sort them by name: 
list = getFileList(in_dir);
list = Array.sort(list);
count = 1
// Set batch mode so that ImageJ won't open these images as they're processed: 
setBatchMode(true);

// Loop over the images in the input folder and process them individually: 
for (i=0; i<list.length; i++)
{
    if (endsWith(list[i], "tif")) //check if its a tif file before processing it
    {
    	open(in_dir+list[i]);
    	id = getImageID(); 
    	title = getTitle(); 
		count = process_img(id, title, n, count);
		selectImage(id); 
		close(); 
    }
}
// Process each input image by slicing it into n patches of 256x256 pixels^2.
function process_img(id, title, n, count){
    getLocationAndSize(locX, locY, sizeW, sizeH); 
    width = getWidth(); 
    height = getHeight(); 
    tileWidth = width / n; 
    tileHeight = height / n; 
    for (y = 0; y < n; y++) { 
        offsetY = y * height / n; 
        for (x = 0; x < n; x++) { 
            offsetX = x * width / n; 
            selectImage(id); 
            call("ij.gui.ImageWindow.setNextLocation", locX + offsetX, locY + offsetY); 
            tileTitle = title + " [" + x + "," + y + "]"; 
            run("Duplicate...", "title=" + tileTitle); 
            makeRectangle(offsetX, offsetY, tileWidth, tileHeight); 
            run("Crop"); 
            savepath = out_dir + var_name + "_" + pad(count,5,0) + ".tif";
            saveAs("Tiff", savepath);
            count = count + 1;
            close(); 
        } 
    } 
    return count;
}
// Helper function to name output patches correctly (0 padding of string). 
function pad (a, left, right) { 
while (lengthOf(""+a)<left) a="0"+a; 
separator="."; 
while (lengthOf(""+separator)<=right) separator=separator+"0"; 
return ""+a; 
} 