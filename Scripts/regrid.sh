#!/usr/bin/env bash
: << 'COMMENT'
'''IMPORTANT!!!!! --> IS APPLIED TO ALL FILES IN DIRECTORY. USE WITH CAUTION'''
COMMENT

get_name () {
    echo $1 | cut -c 10- 
}

for y in $(seq 2020 2020); do
    echo "finding files $y"
    path="./$y"
    files=$(find $path -type f -exec basename {} \;)
    echo "found files $y --> starting regridding procedure"
    for i in $files; do
        echo $i
        # name= $(echo$i | cut -c 10-)
        in_file="$path/$i"
        # echo $in_file
        out_file="$path/$(get_name $i)"
        # echo $out_file
        # $(cdo sellonlatbox,-180,180,-90,90 ifile ofile)
        # $(cdo -O -sellonlatbox,-179.5,179.5,-89.5,89.5 -remapbil,r360x180 $in_file $out_file)
        $(cdo remapbil,grid_layout.txt $in_file $out_file)
        $(chmod ug+rwx $out_file)
        $(rm $in_file) 
        #echo $file
        #echo $start$stop
        #echo $new_file
        done
    echo "finished regridding year $y"
    done
 
