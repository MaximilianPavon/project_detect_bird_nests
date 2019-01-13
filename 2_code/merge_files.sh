for d in */; do
    # Will print */ if no directories are available
    echo $d
    cd $d
    ThermoViewer -exfn result.TMC -merge *.TMC
    ..
done
