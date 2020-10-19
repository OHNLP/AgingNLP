# AgingNLP - Delirium NLP Algorithm

We developed an NLP algorithm to identify patients with delirium from clinical notes.

## Getting Started

Download MedTaggerIE:
https://github.com/OHNLP/MedTagger



### Prerequisites

Java 1.8


### Installing
 
A step by step installation instructions can be accessed through:
https://vimeo.com/392331446

#### NLP framework MedTagger
MedTagger contains a suite of programs that the Mayo Clinic NLP program has developed in 2013.
It includes three major components: MedTagger for indexing based on dictionaries, MedTaggerIE for
information extraction based on patterns, and MedTaggerML for machine learning-based named entity recognition.
#### MedTagger git repo: https://github.com/OHNLP/MedTagger
#### Video demo: https://vimeo.com/392331446
#### Original release: https://github.com/OHNLP/MedTagger/releases

1. Download the latest release from https://github.com/OHNLP/TJA/tree/master/nlp_system 
2. Move the .jar file to either Delirium folder
3. Modify the `INPUTDIR`, `OUTPUTDIR`, and `RULEDIR` variables in `runMedTagger-fit-delirium.sh` or `runMedTagger-fit-delirium.bat`, as appropriate
    - `INPUT_DIR`: full directory path of input folder 
    - `OUTPUT_DIR`: full directory path of output folder
    - `RULES_DIR`: full directory path of 'Rule' folder


## Running the tests
```
runMedTagger-fit-delirium.sh
```

## Built With

* [Maven](https://maven.apache.org/) - Dependency Management

