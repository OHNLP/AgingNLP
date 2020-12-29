#/bin/bash

#change into full directory
INPUT_DIR="/input/"
OUTPUT_DIR="/output/"
RULES_DIR_HY="/HybridEngine"

#Double check the .jar version
java -Xms512M -Xmx2000M -jar MedTagger-fit-1.0.2-SNAPSHOT.jar $INPUT_DIR $OUTPUT_DIR $RULES_DIR_HY