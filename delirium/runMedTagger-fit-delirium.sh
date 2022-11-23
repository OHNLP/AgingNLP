#/bin/bash


INPUT_DIR="/input"
OUTPUT_DIR="/output"
RULES_DIR="/DELIRIUM"

java -Xms512M -Xmx2000M -jar MedTagger-fit-context.jar $INPUT_DIR $OUTPUT_DIR $RULES_DIR


