for d in */; do
  echo $d
  cd $d
  mv frames/* ../frames/
  ..
done
