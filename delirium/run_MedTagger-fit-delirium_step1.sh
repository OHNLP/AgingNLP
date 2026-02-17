#/bin/bash


INPUT_DIR="/delirium/data/input"
OUTPUT_DIR="/delirium/data/output"
RULES_DIR="/delirium/models/ENACT_DELIRIUM"

java -Xms512M -Xmx2000M -jar MedTagger-fit-context.jar $INPUT_DIR $OUTPUT_DIR $RULES_DIR


