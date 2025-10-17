# AgingNLP - Delirium NLP Algorithm

We developed two NLP algorithms (NLP-CAM and NLP-M-CAM) to identify patients with delirium from clinical notes.

## Definition of the NLP-CAM and NLP-mCAM

| A: acute onset and fluctuating course                                                                    | B: inattention                                                                                                                                                                                   |
| -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Do the abnormal behaviors: 1) Come and go, 2) Fluctuate during the day, 3) Increase/decrease in severity | Does the patient: 1) Have difficulty focusing attention, 2) Become easily distracted, 3) Have difficulty keeping track of what is said                                                           |
| C: disorganized thinking                                                                                 | D: altered level of consciousness                                                                                                                                                                |
| Is the patient’s thinking: 1) Disorganized, 2) Incoherent                                                | What is the patient’s level of consciousness: 1) Alert (normal), 2) Vigilant (hyper-alert), 3) Lethargic (drowsy but easily roused), 4) Stuporous (difficult to rouse), 5) Comatose (unrousable) |
| Original CAM:                                                                                            | Modified CAM:                                                                                                                                                                                    |
| Definitive: A and B and (C or D)                                                                         | Definitive: At least 3 unique CAM criteria; Possible: Any 2 criteria and does not meet the definitive criteria as above                                                                          |

## Definition of the CAM-related Clinical Concepts

| Delirium Concepts     | Example Terms                                 | Primary Indication of CAM | Require Additional Context | Potential Direct Indication of Delirium Status |
| --------------------- | --------------------------------------------- | ------------------------- | -------------------------- | ---------------------------------------------- |
| Agitation             | agitated, agitation                           | CAM D                     | Yes                        | No                                             |
| AMS                   | mental status change                          | CAM A                     |                            | No                                             |
| Confusion             | confusion, confused                           | CAM C                     | Yes                        | No                                             |
| Delirium              | delirious, delirium                           | Delirium                  |                            | Yes                                            |
| Disconnected          | disconnected                                  | CAM C                     |                            | No                                             |
| Disorganized_thinking | jumped from topic to topic, paranoid thoughts | CAM C                     |                            | No                                             |
| Disorient             | impaired orientation                          | CAM C                     |                            | No                                             |
| Drowsy                | drowsy                                        | CAM D                     |                            | No                                             |
| Encephalopathy        | encephalopathy, leukoencephalopathy,          | Delirium                  |                            | Yes                                            |
| Fluctuation           | fluctuation, night and day different          | CAM A                     |                            | No                                             |
| Hallucination         | hallucination, seeing things                  | CAM C                     |                            | No                                             |
| Inattention           | inattentive, not paying attention             | CAM B                     |                            | No                                             |
| Reorient              | reorientation, reorientated                   | CAM C                     | Yes                        | No                                             |

## Getting Started

### Prerequisites

Java 1.8

#### NLP framework MedTaggerIE

MedTagger contains a suite of programs that the Mayo Clinic NLP program has developed in 2013. It includes three major components: MedTagger for indexing based on dictionaries, MedTaggerIE for information extraction based on patterns, and MedTaggerML for machine learning-based named entity recognition.

### Download

#### https://github.com/OHNLP/AgingNLP/tree/master/delirium

### Install

A step by step installation instructions can be accessed through:
https://vimeo.com/392331446

1. Move the .jar file to the Delirium folder
2. Modify the `INPUTDIR`, `OUTPUTDIR`, and `RULEDIR` variables in `runMedTagger-fit-delirium.sh` or `runMedTagger-fit-delirium.bat`, as appropriate
   - `INPUT_DIR`: full directory path of input folder
   - `OUTPUT_DIR`: full directory path of output folder
   - `RULES_DIR`: full directory path of 'Rule' folder

### Run

```
runMedTagger-fit-delirium.sh
```

### Refine

Due to the institutional-specific heterogeneity, after deplying the algorithm, we recommend 1) conducting local evaluation through manual chart review, 2) refining the keywords, 3) implementing a section detection algorithm based on the structure of clinical notes. Additional information the NLP deployment and evaluation process can be found at: https://github.com/OHNLP/annotation-best-practices

## Reference

Fu S, Lopes GS, Pagali SR, Thorsteinsdottir B, LeBrasseur NK, Wen A, Liu H, Rocca WA, Olson JE, St Sauver J, Sohn S. Ascertainment of delirium status using natural language processing from electronic health records. The Journals of Gerontology: Series A. 2020 Oct 30.
