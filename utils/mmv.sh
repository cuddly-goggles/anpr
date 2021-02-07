#!/bin/bash

for f in alldataset/2/*.jpg
do
  echo "${f%.jpg}.txt"
  mv -- "$f" "${f%.jpg}c.jpg"
  touch "${f%.jpg}c.txt"
done