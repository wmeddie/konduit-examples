#!/usr/bin/env bash

for i in {1..100}
do
    curl -F default=@1902_airplane.png http://localhost:1337/json/image > /dev/null &
done

curl -F default=@1902_airplane.png http://localhost:1337/json/image

echo "done"