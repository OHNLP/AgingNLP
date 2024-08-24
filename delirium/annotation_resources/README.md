**Delirium Study Annotation Guideline**

Revision 1.8

Date: 05/15/2019

# 1. Background

The REP contains extensive, structured data (e.g., ICD-9 and ICD-10 codes). The unstructured EHR clinical notes also contain a wealth of information on symptoms, diseases, and syndromes. However, it is time-consuming and costly to manually extract information from clinical records for large patient populations. To address this problem, several groups of investigators, including our own, have used NLP algorithms to identify clinical conditions and biomedical concepts from radiology reports, discharge summaries, problem lists, nursing documentation, and medical education documents. In the current study, we will partner with experts in Medical Informatics to mine EHRs using NLP techniques to identify the geriatric syndrome "delirium."

In the current study, we aim to identify patients diagnosed with "delirium." Delirium is not routinely coded, and a study of patients undergoing elective surgery indicated that delirium was given an ICD-9 code in only 3% of patient records. Our goal is to develop, benchmark, and evaluate NLP algorithms to identify the geriatric syndrome "delirium" from unstructured Mayo Clinic clinical notes (Mayo Clinic EHR). These NLP techniques may be adapted and broadly applied to other geriatric syndromes.

# 2. Annotation tool

The annotation tool for this project is Multi-document Annotation Environment (MAE), a Java-based natural language annotation software package. MAE is a non-web-based annotation tool. All annotation tasks were defined in a document type definition (DTD) file. Due to its lightweight feature, the software can be easily shared and updated across multiple sites without configuration and testing.

Download: [https://github.com/keighrim/mae-annotation](https://github.com/keighrim/mae-annotation)

![](RackMultipart20230712-1-jaoqzd_html_b623d5af8df7cd98.png)

# 3. Instructions

Please use this document for annotation to make sure we are on the same page. The majority of the information will be in the method section. Be sure to check the supplementary materials as well.

**Bottom line:** Annotators will be given a list of medical records. For each medical record, annotators will identify and highlight keywords. Then, annotators will assign attributes to these keywords. Finally, annotators will assign a definition of delirium that seems most appropriate for the patient. All this must be done using the software MAE.

**Training:** Each annotator will be given an initial 1h training session and ask to annotate two medical records from the initial guidelines development set and to then compare their annotations with the gold standard. Questions raised from the training exercise will be used to refine the baseline guideline.

**Annotation Flow:**

Guideline Development Phase: Each annotator will annotate a batch of clinical notes. Then, inter-annotator agreement (IAA) will be calculated and the guidelines will be revised. The process continues until a high agreement is reached.

Annotation Phase: **300** clinical notes will be annotated. After the annotations are completed, we computed IAA, resolved disagreements, and clarified the guidelines. Each document is independently annotated by 2 annotators.

Annotators will assign one definition to each patient (i.e., definitive, possible, no delirium; see item 4.2). The annotation stops when all medical records of that patient were examined.

Please, record the time spending annotating each document. Please, take your time and try your best to annotate all concepts and properties.

Adjudication: The final gold standard annotations will be created by combining the individual experts' annotations followed by adjudication of the mismatches. The lessons learned and case examples from the 22 clinical notes used for training were added to the annotation guideline.

# 4. Guideline

When annotators open the medical record file, they will notice that most keywords and their variations will be already highlighted. These pre-highlighted keywords must serve only as a guide, as **by no means** the annotator should limit the annotation to these keywords.

Annotators will have four responsibilities for each medical record. Annotators will:

(1) Search for and highlight expressions that are not already highlighted, but are conceptually similar to a keyword;

(2) Revise the highlighted keywords;

(3) Assign attributes to each highlighted expression;

(4) Assign a definition of delirium that seems most appropriate for the patient.

## 4.1 Search for and highlight expressions that are not already highlighted, but are conceptually similar to a keyword;

Annotators will search for expressions that are conceptually similar to the following keywords throughout the medical records:

Delirium

CAM Positive

AMS

Hallucination

Confusion

Disoriented

Reoriented

Encephalopathy

Agitated

Fluctuation

Inattention

Disorganized thinking

Disconnected

Definition (although this is on the keyword list, this is **not** a keyword; see item 4.4)

Most keywords and their variations will be already highlighted. Annotators must search for expressions that are not yet highlighted, but that are conceptually associated with a keyword. Annotators must only highlight expressions that can be assigned a CAM criterion (see item 4.3).

For example, expression such as "fuzzy thoughts" and "nonsensical speaking" should be assigned to the keyword "disorganized thinking" (and also to the CAM criterion A; see item 4.3). Expressions such as "slow to respond", "drowsy but easily arousable", "decreased responsiveness", "awake but minimally responsive", and "increased lethargy" should be assigned to the keyword "disconnected" (and also to the CAM criterion D; see item 4.3).

## 4.2 Revise the highlighted keywords

When annotators open the file, they will notice that most keywords and their variations will be already highlighted. These pre-highlighted keywords must serve as a guide. If the keyword does not have a context associated with it, then the annotator must "un-highlight" the keyword. If there is a context, then annotators must "un-highlight" the keyword and then highlight the entire sentence to that keyword (and respective CAM criterion).

**Example (1)**: The annotator opens the file, and the following keyword is highlighted to confusion: "_diagnosis:_ _confusion_". We cannot assign CAM criterion A to this expression because there is no context associated with it. Therefore, the annotator simply "un-highlights" this keyword.

**Example (2)**: The annotator opens the file, and the following keyword is highlighted to confusion: "_pt has_ _confusion_ _but it fluctuates during the day_". Given the context, we can assign CAM criterion A to this expression because criterion A refers to fluctuations of a symptom or condition. Therefore, the annotator must "un-highlight" the keyword and highlight the entire expression "_pt has_ _confusion but it fluctuates during the day_", and then assign CAM criterion A to it.

## 4.3 Assign attributes to each highlighted keyword

For each highlighted expression, annotators must fill a set of attributes about that keyword. Each keyword has 5 or 6 attributes, and each attribute has categories that must be selected. Examples of attributes are within the red square in the picture.

Below is the list of attributes and associated categories in parenthesis.

**CAM Criteria (A/B/C/D/Multiple/Indeterminate):** For each expression, annotators should assign whether it is associated with a particular CAM criteria. If there is no CAM criterion to be selected (i.e., "Indeterminate"), then the annotator does NOT need to highlight the expression at all. All keywords have this attribute, with the exception of "Delirium" and "CAM Positive".

The context within which an expression is used may reveal information about whether that expression is associated with a certain CAM criterion. For example, one medical record says "_patient experienced a_ _confusion_ _episode_". This expression by itself does not meet any CAM criteria. However, this sentence was used within a context—"_patient experienced a_ _confusion_ _episode and described that she was feeling very drowsy and therefore could not think straight_". Now, the expression "confusion episode" may be associated with the CAM criterion "D" (altered level of consciousness) and "C" (disorganized thinking), for example. The annotator must therefore highlight the entire sentence "_patient experienced a_ _confusion episode and described that she was feeling very drowsy and therefore could not think straight_" and assign CAM Criteria "Multiple". The context should be highlighted together with the expression.

Examples of contexts that allow for assigning a CAM criterion to a keyword:

**CAM A**** : Acute onset/Fluctuating course**

**Expression** : "_pt says that __**AMS**__ comes and goes_"

**Annotator** : assign the entire expression "_pt says that__ **AMS** _ _comes and goes_" to the keyword "AMS" and assign CAM criterion A_._

**Expression** :"_patient is sometimes_ _ **disoriented** __; last week it happened three times as per patient's daughter_"

**Annotator** :assign the entire expression "_patient is_ _sometimes_ _ **disoriented** __;_ _last week it happened three times_ _as per patient's daughter_" to the keyword "disoriented" and assign CAM criterion A.

**Expression** :"_pt:_ _ **disoriented** _"

**Annotator** :this expression does not provide any context for CAM, so it must be "un-highlighted".

**Expression** :"_It is important to note that the patient's_ _ **mental status** __has been_ _ **fluctuating**__ over the last few days_"

**Annotator** :assign the expression "_patient's_ _ **mental status** __has been_ _ **fluctuating**__ over the last few days_" to the keyword "AMS" and assign CAM criterion A_._

**Expression** : "_Insight and judgment were currently intact as evidenced by her understanding of the situation, however has clearly_ _ **fluctuated** _ _during this hospital stay_"

**Annotator** :assign the entire expression to "AMS" and assign CAM criterion A.

**Note** : The presence of a keyword overrides the keyword "Fluctuation". That is, in cases the context includes information about a keyword together with fluctuation, the annotator should NOT assign it to the keyword "Fluctuation". Instead, the annotator should assign it to the keyword that represents the expression (for example, "_fluctuating_ _hallucinations_" should be assigned to "Hallucinations" and then to the CAM criterion A). The keyword "Fluctuation" should only be assigned in very rare cases where "fluctuation" is the only specific information in the expression (for example, "_pt cognitive symptoms are_ _fluctuating_").

**CAM B**** : Inattention**

Because CAM B refers directly to inattention, the keyword "inattention" usually does not require much context to be assigned a CAM B. See examples below:

**Expression** : "_I could not communicate with patient because he was having difficulty focusing_ _ **attention** _"

**Annotator** : Just leave it highlighted and assign CAM criterion B_._

**Expression** : "_[…] concerning_ _ **inattention** _ _to surroundings_"

**Annotator** : Just leave as it is and assign it to CAM criterion B (of course, always check the context).

**Expression** : "_The_ _ **inattention** _ _of pt to my instructions was due to him using his cellphone while I was speaking_"

**Annotator** : In this case, the context indicates that the inattention was not a symptom or condition. The annotator must "un-highlight" the expression if it is pre-highlighted, then highlight the entire expression "_The_ _ **inattention** __of pt to my instructions was due to him using his cellphone while I was speaking_", and then assign "Negated".

**CAM C**** : Disorganized thinking**

**Expression** : "_Patient has_ _slurred speech_ _/_ _garbled speech_ _/_ _changes in speech pattern_"

**Annotator** : Just leave it highlighted and assign CAM criterion C_._

**Expression** : "_This morning patient showed_ _unclear flow of ideas_ _when describing the episode_"

**Annotator** : Highlight this expression if not already highlighted, and assign CAM criterion C to it_._

**CAM D**** : Altered level of consciousness**

**Expression** : "_Patient is also lethargic this morning due to_ _ **disorientation** _"

**Annotator** : Highlight the entire expression"_Patient is also lethargic this morning due to_ _ **disorientation** _" and assign it to CAM criterion D.

**Expression** : "_Patient has shown_ _decreased responsiveness_ _this morning_"

**Annotator** : Assign this expression to the keyword "Disconnected" and then assign it to CAM criterion D.

**NOTE:** Expressions such as "slow to respond", "drowsy but easily arousable", "decreased responsiveness", "awake but minimally responsive", and "increased lethargy" should be assigned to the keyword "disconnected" and to the CAM criterion D.

Some keywords or expressions are more easily associated with one or more CAM criteria. The annotator needs to pay special attention to them. The list below can give a rough idea of what to expect of each keyword, but please pay attention to the unique context (if there is any) in which the keyword is being used:

**Delirium** (not associated with any particular CAM criterion)

**CAM Positive** (not associated with any particular CAM criterion)

**AMS** (easily associated with criteria A, B, and mostly C)

**Hallucination** (easily associated with criteria A)

**Confusion** (easily associated with criteria A, B, C, or D; it really depends on the context)

**Disoriented** (easily associated with criteria A, B, or C)

**Reoriented** (not usually associated with any particular CAM criterion)

**Encephalopathy** (not usually associated with any particular CAM criterion)

**Agitated** (easily associated with criterion A)

**Fluctuation** (easily associated with criterion A)

**Inattention** (strongly associated with criterion B)

**Disorganized thinking** (strongly associated with criterion C)

**Disconnected** (strongly associated with criteria A and sometimes D)

**Certainty (Confirmed/Hypothetical/Possible/Negated):** How certain was the healthcare provider when using a keyword?

Confirmed: The keyword was used in a very clear and positive manner.

_Patient_ _has_ _delirium_

_He_ _has_ _history of hallucinations_

_She_ _is_ _CAM Positive_

Hypothetical: The note asserts that the patient may develop a medical problem.

_If_ _you experience disorientation_

_In case_ _you feel confused again_

Possible: Patient may have a symptom or condition, but there is uncertainty expressed in the note.

_This is very_ _likely / unlikely_ _to be an episode of delirium._

_Doctors_ _suspect_ _altered mentation._

_Questionable_ _/ small chance of hallucinations._

_Disorientation is_ _possible_ _/_ _probable__._

_Suspicion_ _of AMS._

_We are_ _unable to determine_ _whether she has hallucinations_.

_It is_ _possible / likely / thought / unlikely_ _that she has delirium._

_We_ _suspect_ _this is not encephalopathy._

_This is_ _probably_ _not hallucinations._

Negated: The keyword did not happen or is said to not exist.

_Patient_ _does not_ _have encephalopathy._

_No_ _history of confusion_

**Note 1** : "Negated" must be assigned if the expression was used in the correct context, but in a negated form (see examples above). In contrast, the attribute Exclusion must be assigned "Yes" if the expression is completely out of context (see item 5 "Exclusion" below).

**Note 2** : In case the expression is "Negated", the annotator must include the negated sentence in the highlighted expression. For example, "_pt was not_ _confused_" must be highlighted as "_pt was_ _not confused_" and then assign "Negated".

**Note 3** : Also, "Possible" takes precedence over "Negated". Therefore, annotators should categorize terms like "probably not" or "unlikely" as being "Possible".

**Status (History/Present):** Is the keyword being used in the context of past experiences or present circumstances (i.e., within a week)?

History

_Patient has_ _history_ _of hallucinations_

_His father has_ _history_ _of AMS_

_Patient described episodes of delirium_ _several years ago_

_His hallucinations were_ _resolved_

Present

_Patient_ _has_ _encephalopathy_

_Patient experienced episode of confusion_ _this morning_

_Currently_ _CAM Positive_

_She has experienced confusion several times during the_ _last few days_

**Experiencer (Patient/Other):** To whom is the keyword referring?

Patient

_Patient_ _had delirium this morning_

_Patient_ _has history of AMS_

Other

_His father_ _has history of hallucinations_

**Exclusion** **(Yes/No) and Comments**:Annotators should select "Yes" if the expression is being used in a completely different context. For example, "_the nurse_ _disconnected_ _the call_" or "_pt was_ _confused_ _was to whether the appointment was at Gonda or St Marys_". Please, elaborate in the comments if needed. Please, feel free to comment anything that could be useful information.

## 4.4 Assign a definition of delirium that seems most appropriate for the patient

After revising and highlighting all the keywords and expressions, and after assigning attributes to each expression, annotators will assign a definition of delirium that seems most appropriate for the patient. The annotator's decision should be based on the information contained in the patient's medical records **as a whole**.

Annotators will use the keyword "Definition" to assign a definition. Annotators will simply highlight a random sentence to allow the categories to be selected, and then select the appropriate category. The keyword "Definition" has one attribute with the following categories: Definitive / Possible / No delirium.

Our definition of delirium is primarily based on the physician's diagnosis. In ca ses that there is no clear diagnosis of delirium, then the definition is the criteria recommended by the Confusion Assessment Method – CAM (see below).

**Definitive delirium** : **(1)** The medical records explicitly mention that the patient **definitely** experienced an episode of delirium, **AND/OR** **(2)** The medical records describe symptoms that match criteria A and B and either C or D of the CAM **and** these symptoms were experienced within a week.

For example, the diagnosis of a patient who experienced acute altered mental status (CAM A), inattention (CAM B), and disorganized thinking (CAM C), and these symptoms occurred within one week, is considered "definitive". A patient whose medical records explicitly mention a clear episode of delirium is also considered "definitive" (e.g., "_#1 Delirium, multifactorial, resolving_", "_#4 Acute post-operative delirium_", and "_Patient is CAM positive_"), even if there are no descriptions of occurrence of delirium-related symptoms elsewhere in the medical records.

**Possible delirium** : **(1)** The medical records explicitly mention that the patient **probably/possibly** experienced an episode of delirium, **AND/OR** **(2)** The medical records describe symptoms that match one or more than one CAM criterion, but less than what is necessary for "definitive delirium".

For example, the diagnosis of a patient who experienced acute altered mental status (CAM A) and inattention (CAM B), but has **not** experienced either disorganized thinking (CAM C) or altered level of consciousness (CAM D) is considered "possible." A patient whose medical records explicitly mention a probable or possible episode of delirium is considered "possible" (e.g., "_The patient does appear to exhibit a mild delirium_", "_Likely represents post-operative delirium_", "_Unwitnessed fall, altered mental status, delirium ?_").

**No delirium** : The medical record does not meet the criteria for "Definitive" or "Possible" described above.
