#!/bin/bash

cd /Users/alex/Desktop/Sphinx || exit
git add .
git commit -m "Automated commit $(date +'%Y-%m-%d %H:%M:%S')"
git push origin main
