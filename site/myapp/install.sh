#!/bin/bash

sudo cp plant-detector.service /etc/systemd/system/

sudo chmod 644 /etc/systemd/system/plant-detector.service

sudo systemctl enable plant-detector

sudo systemctl start plant-detector


