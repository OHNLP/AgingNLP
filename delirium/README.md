# AgingNLP - Delirium NLP Algorithm

We developed an NLP algorithm to identify patients with delirium from clinical notes.

## Definition of the NLP-CAM

| A: acute onset and fluctuating course | B: inattention |
| --- | --- |
| Do the abnormal behaviors?
 - Come and go
 - Fluctuate during the day
 - Increase/decrease in severity | Does the patient:  - Come and go
 - Fluctuate during the day
 - Increase/decrease in severity |


| Delirium Concepts | Example terms | Primary indication of CAM | Require additional context | Potential indication of delirium status |
| --- | --- |
| Agit | agitated, agitation | CAM D | Yes | No |
| AMS  | mental status change | CAM D |  | No |
| Confusion  | confusion, confused | CAM C or CAM D | Yes | No |
| Delirium | delirious, delirium | Delirium equivalent |  | Yes |
| Disconnected  | disconnected | CAM C |  | No |
| Disorganized_thinking  | jumped from topic to topic, paranoid thoughts | CAM C |  | No |
| Disorient  | impaired orientation | CAM C |  | No |
| Drowsy | drowsy | CAM D |  | No |
| Encephalopathy  | encephalopathy, leukoencephalopathy,  | CAM D |  | Yes |
| Fluctuation | fluctuation, night and day different | CAM A |  | No |
| Hallucination  | hallucination, seeing things | CAM C  |  | No |
| Inattention  | inattentive, not paying attention | CAM B |  | No |
| Reorient  | reorientation, reorientated | CAM C | Yes | No |

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
2. Move the .jar file to the Delirium folder
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


## Citation
Fu S, Lopes GS, Pagali SR, Thorsteinsdottir B, LeBrasseur NK, Wen A, Liu H, Rocca WA, Olson JE, St Sauver J, Sohn S. Ascertainment of delirium status using natural language processing from electronic health records. The Journals of Gerontology: Series A. 2020 Oct 30.

