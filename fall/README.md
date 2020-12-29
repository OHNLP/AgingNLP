# AgingNLP - Fall NLP Algorithm

We developed an NLP algorithm to identify fall occurrence from clinical notes.

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
2. Move the .jar file to Fall folder
3. Modify the `INPUTDIR`, `OUTPUTDIR`, and `RULEDIR` variables in `runMedTagger-fit-delirium.sh` or `runMedTagger-fit-delirium.bat`, as appropriate
    - `INPUT_DIR`: full directory path of input folder 
    - `OUTPUT_DIR`: full directory path of output folder
    - `RULES_DIR`: full directory path of 'Rule' folder


## Running Hybrid Model
### Step1
Train BERT on training data:
output_dir: directory to save post trained BERT model
Input training file: training.txt
Input data format: 'docid', 'sentence', 'label'
delimiter: '|'
```
python bert/main.py -train
```
### Step2
Output prediction results
output_dir: directory to save post trained BERT model
test_dir: input test file
```
python bert/main.py -reload
```
### Step3
Run MedTagger Summariation Engine

#### CONFIGURATION:
INPUT_DIR: full directory path of input folder
OUTPUT_DIR: full directory path of output folder
RULES_DIR: full directory path of 'Hybrid' folder

###### INPUT:
 Input folder: the input folder contains a list of clinical notes 
 Input file: document level .txt file. The naming convention of each report would be unique identifier + documentation date. P.S. one patient may have multiple documents. 
 Input file preprocessing: replace all '/n' to '. '

```
runMedTagger-fit-hybrid.sh
```



