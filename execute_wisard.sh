#!/bin/bash

t=2
while [ $t -lt 18 ]
do
   echo "iniciando o threshold " $t
   python src/main.py --threshold $t
   if [ $t -eq 18 ]
   then
      break
   fi
   t=`expr $t + 1`
done
