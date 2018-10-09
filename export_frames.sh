for d in */; do
    # d = directory
    echo $d
    cd $d
    mkdir -p frames
    i=0
    for file in `ls $1`; do
      if [ $file = "frames" ]; then
      else
        # just for testing purposes:
        # ThermoViewer -i $file -cp iron -expa frames/ -exfn "${d//\//}_$i" -exfo jpg -exsf 12 -exef 12 -c
        ThermoViewer -i $file -cp iron -expa frames/ -exfn "${d//\//}_$i" -exfo jpg -c
      fi
      ((++i))
    done
    ..
done
