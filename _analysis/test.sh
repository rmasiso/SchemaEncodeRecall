#!/bin/bash

rois=( x 1 )
if [ ${#rois[@]} -eq 2 ]; then
    echo "They are equal!"
else
    echo "not sure"
fi
