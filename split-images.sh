#!/usr/bin/bash

#Splits images of T200 and T500 data into 36x36 images (so the autoencoder can handle it...)
#Requires ImageMagick

for i in $(seq 1 3000);do
	magick ./image_3D_png/T200/img3d_$i\_T200.png -crop 360x360+62+22 ./image_3D_png_tiled/temp/crop1.png
	magick ./image_3D_png_tiled/temp/crop1.png -crop 10x10@ ./image_3D_png_tiled/T200/img3d_$i\_%02d.png
	rm ./image_3D_png_tiled/temp/crop1.png

	magick ./image_3D_png/T500/img3d_$i\_T500.png -crop 360x360+62+22 ./image_3D_png_tiled/temp/crop1.png
	magick ./image_3D_png_tiled/temp/crop1.png -crop 10x10@ ./image_3D_png_tiled/T500/img3d_$i\_%02d.png
	rm ./image_3D_png_tiled/temp/crop1.png
	echo $i
done;
