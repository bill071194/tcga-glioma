# **Classification of Glioma Gradings Based on Genetic Mutations and Population Characteristics**

## Table of Contents

[INTRODUCTION](#_Toc141866606)

[DATASET](#_Toc141866607)

[DATA ANALYSIS](#_Toc141866608)

[LOGISTIC REGRESSION RESULTS](#_Toc141866609)

[DISCUSSION](#_Toc141866610)

[REFERENCES](#_Toc141866611)

[APPENDIX](#_Toc141866612)

## **INTRODUCTION**

Glioma is a known type of cancer where cells grow abnormally in the brain or spinal cord. The malignant cells in particular, expand swiftly and can penetrate and destroy healthy tissues in affected areas. Among all malignant glioma categories, Glioblastoma Multiforme (GBM) is regarded as the most malignant and has the worst prognosis (Shergalis et al., 2017). GBM patients were found to have a median survival term of 14-month and an average 6% 5-year survival rate, the lowest among all categories of malignant brain tumors (Thakkar et al., 2014; Shergalis et al., 2017). The astrocytes that support nerve cells are found to be the source of glioblastoma development (Mayo Clinic, 2019). In addition to its malignant nature, GBM also affects individuals and health care systems significantly in terms of the cost of care. The increased costs of GBM treatments are a result of invasive surgical interventions and rare pharmacokinetic interventions such as the use of temozolomide (TMZ) and other antibody drugs, which incur significant costs (Shergalis et al., 2017; Jiang et al. 2017). According to Jiang et al., the average total costs per patient following resection surgery ranged from $136,787 to $184,107 (Jiang et al., 2017). In a study by Aly et al., in "_Survival, costs, and health care resource use by line of therapy in US Medicare patients with newly diagnosed glioblastoma: a retrospective observational study_", it was also discovered that these costs do not stop shortly after these tertiary interventions, but also further down the line for individuals with GBM recurrence (Aly et al., 2020). These detrimental effects of GBM on the health care system and individuals created a compelling need to discover other methods for early detection and classification of the tumor.

While 80% of GBM are comprised of primary GBM, which occurs spontaneously through a myriad of factors such as genetics and age, the rest of the category consists of secondary GBM, which develops from lower-grade astrocytoma, or lower grade gliomas (LGG) (Thakkar et al., 2014). An efficient and timely methodology of classifying LGG and GBM based on biomarkers has not been widely covered in the existing literature body. In various studies by Shergalis et al. in "_Current Challenges and Opportunities in Treating Glioblastoma_", Disney-Hogg et al. in "Influence of obesity-related risk factors in the aetiology of glioma", and Wiedmann et. al. in "_The impact of body mass index and height on the risk for glioblastoma and other glioma subgroups: a large prospective cohort study_" respectively, the investigators attempted to associate other factors such as lifestyle and diet to GBM; however, their respective findings remain sporadic and inconclusive (Shergalis et al., 2017; Disney-Hogg et al., 2018; Wiedmann et al., 2017). One of the most promising potential factors to classify GBM currently is using bioinformatics to explore the effect of genomes and genetic mutations and the role they play in the development of GBM (Shergalis et al., 2017). Shergalis et al., in a study by The Cancer Genome Atlas Project, stated that 20 genes overexpressed in a subpopulation of GBM include: BOC, CLEC4GP1, ELOVL6, EREG, ESR2, FDCSP, FURIN, FUT8-AS1, GZMB, IRX3, LITAF, NDEL1, NKX3-1, PODNL1, PTPRN, QSOX1, SEMA4F, TH, VEGFC, and C20orf166AS1 (Shergalis et al., 2017). Using this precedent, this project aims to leverage genomic data in the classification of GBM using 20 other genetic mutations in a dataset provided by The Cancer Genome Atlas Project.

**Objective**

This paper aims to explain and classify Glioblastoma Multiforme and lower grade gliomas using genetic mutations and population characteristics as features in a logistic regression machine learning model.


## **DATASET**

The dataset being used for this project came from The Cancer Genome Atlas Project. The dataset comprises _27 attributes_ and _844 instances_ (Tasci et al., 2022). The target variable is the _Glioma grade class information (0: LGG, 1: GBM)._ The data dictionary for the attributes is illustrated below:

| Attribute | Description | Dummy Variables |
| --- | --- | --- |
| Sex | Sex of the recorded case | 0: Male1: Female |
| Age | Age of case at diagnosis. Calculated number of days. | Continuous variable |
| Race | Race of recorded case | Race\_whiteRace\_asianRace\_blackRace\_native american or alaska native |
| IDH1 | isocitrate dehydrogenase (NADP (+))1 | 0: Not mutated1: Mutated |
| TP53 | tumor protein p53 | 0: Not mutated1: Mutated |
| ATRX | ATRX chromatin remodeler | 0: Not mutated1: Mutated |
| PTEN | phosphatase and tensin homolog | 0: Not mutated1: Mutated |
| EGFR | epidermal growth factor receptor | 0: Not mutated1: Mutated |
| CIC | capicua transcriptional repressor | 0: Not mutated1: Mutated |
| MUC16 | mucin 16, cell surface associated | 0: Not mutated1: Mutated |
| PIK3CA | phosphatidylinositol-4,5-bisphosphate 3-kinase catalytic subunit alpha | 0: Not mutated1: Mutated |
| NF1 | neurofibromin | 0: Not mutated1: Mutated |
| PIK3R1 | phosphoinositide-3-kinase regulatory subunit 1 | 0: Not mutated1: Mutated |
| FUBP1 | far upstream element binding protein 1 | 0: Not mutated1: Mutated |
| RB1 | RB transcriptional corepressor 1 | 0: Not mutated1: Mutated |
| NOTCH1 | notch receptor 1 | 0: Not mutated1: Mutated |
| BCOR | BCL6 corepressor | 0: Not mutated1: Mutated |
| CSMD3 | CUB and Sushi multiple domains 3 | 0: Not mutated1: Mutated |
| SMARCA4 | SWI/SNF related, matrix associated, actin dependent regulator of chromatin, subfamily a, member 4 | 0: Not mutated1: Mutated |
| GRIN2A | glutamate ionotropic receptor NMDA type subunit 2A | 0: Not mutated1: Mutated |
| IDH2 | isocitrate dehydrogenase (NADP (+)) 2 | 0: Not mutated1: Mutated |
| FAT4 | FAT atypical cadherin 4 | 0: Not mutated1: Mutated |
| PDGFRA | platelet-derived growth factor receptor alpha | 0: Not mutated1: Mutated |

**Dataset URL**

[https://archive.ics.uci.edu/dataset/759/glioma+grading+clinical+and+mutation+features+dataset](https://archive.ics.uci.edu/dataset/759/glioma+grading+clinical+and+mutation+features+dataset)

## **DATA ANALYSIS**

**Feasible Approaches**

Given the dichotomized nature of the outcome variable, there are several approaches for analytics in this project. Below are two notable ones:

- Supervised Learning (Classification): A machine learning model could be used to classify whether the target is GBM or LGG. The given dataset will be split into 70% training data and 30% test data. The analysis will pick the most popular classification algorithms and compare the accuracy scores when predicting the test data set using the training model. An ensemble method such as boosting will also be considered depending on the results of the machine learning model.
- Logistic Regression: A logistic regression model could be used to explain and predict the target. The dataset will go through tests for multicollinearity (VIF) and auto-correlation (Durbin Watson) prior to the logistic regression analysis to identify any confounding factors. Afterwards, the selected attributes and instances will be used to fit a logistic regression model. Attributes with significant p-values and test statistics will be selected to explain the model and provide clinical insights in explaining the target variable (presence of GBM compared to LGG).

**Project Analysis**
 In this project, a hybrid method of both approaches will be utilized to identify features that can explain and classify glioblastoma multiforme (GBM) against lower grade gliomas (LGG). The analysis steps are illustrated in the flow chart below (Appendix Figure 1).

Tools and Libraries

This analysis will be conducted using the R Project for Statistical Computing version 4.3.1 from CRAN, and the RStudio Version 2023.06.0 interface for R. The libraries and packages used for the analysis include _dplyr, stringr, fastDummies, ggcorrplot, ggstatsplot, Boruta, caret, mlbench, car, ada, naivebayes,_ and _e1071_.

Data Preprocessing

After importing the original data, the data goes through a series of data preprocessing techniques before it is used for analysis. The data preprocessing process involves removing 5 instances with index _42, 438, 672, 707,_ and _795_ due to missing values. Additionally, 3 attributes were removed for not contributing to the model, which are _Project, Case\_ID_, and _Primary\_Diagnosis_. Next, the _Age\_at\_diagnosis_ attribute is string split and converted from the "X years Y days" string format to a numeric double value specifying age in the unit of years. Next, the categorical variables are converted into dummy variables using the _fastDummies_ package in R, with reference attributes dropped to prevent redundancy and multicollinearity. Finally, the target variable, _Grade\_GBM_, is converted into factor for classification purpose. All columns were then combined and renamed for analysis and ease of interpretation. The final data used for analysis consists of 839 instances, 25 feature variables, and 1 target variable.

Exploratory Data Analysis (EDA)

Before using the final data set to do analysis, a histogram of the target classes is used to determine if there were over-represented or under-represented classes (Appendix Figure 2). The histogram analysis indicates that there are 487 instances with LGG and 352 instances with GBM. The difference between the 2 classes is not significant enough to conclude if there exists any over-representation. Next, a correlation matrix using Pearson correlation is used to determine if there are any autocorrelations between the feature variables (Appendix Figure 3). The results suggested that no strong correlations exist, with all correlation coefficients being \<0.7.

Holdout Approach

To prevent overfitting, the data was split into a training data set, used for the explanatory model and train classification models, and a testing data set, used for predictions and measuring prediction accuracy. The data was split in a 70:30 ratio, with 70% of the data (587 instances) as training data and 30% of the data (252 instances) as testing data.

Feature Selection
 To select only features with the most importance to the target variable, 2 packages were used: Boruta and caret Recursive Feature Elimination. Both of the feature selection techniques used the random forest algorithm to calculate feature importance, with a few differences. Boruta employs variable shadows to generate recommendations on which variables are most important, while RFE goes through a series of recursive tests to select the set of variables that would yield the highest accuracy for the model. In this scenario, Boruta suggests that the attributes that are important to the target variable (GBM presence) are: _IDH1, Age, ATRX, CIC, IDH2, EGFR, PIK3R1, PTEN, NOTCH1, RB1, FUBP1, SMARCA4, TP53, PDGFRA_, and _GRIN2A_ (Appendix Figure 4)_._ RFE on the other hand, suggests that having _IDH1, IDH2, Age, CIC, ATRX, PIK3R1, PTEN, RB1, EGFR, GRIN2A, TP53, FUBP1, NOTCH1, SMARCA4, CSMD3, MUC16, Race\_black, Race\_asian, PIK3CA, Race\_native\_aa, FAT4, BCOR,_ and _NF1_ included in the model would yield the highest accuracy (Appendix Figure 5)_._

Logistic Regression

Using variables recommended from the feature selection techniques above, 2 logistic regression models were developed using training data with only features selected by each technique. After comparing both the Chi Square statistics and prediction accuracy, the features selected by the RFE provided better fit and accuracy, which leads the analysis to use these variables for additional analyses.

Training Classification Algorithms

To compare with the logistic regression model, the RFE training data was used to train other popular classification models, include Naïve Bayes, Adaptive Boosting (Adaboost), Random Forest, Linear Support Vector Machine, and Non-Linear Support Vector Machine. The accuracy comparison is illustrated in Figure 6 (Appendix Figure 6). With the highest accuracy score of **0.8968** , the analysis will pick the Logistic Regression model as the predictor model for GBM classification.


## **LOGISTIC REGRESSION RESULTS**

Overall, the logistic regression model yielded a Chi square value of 431.64 and a p-value of 0.000, indicating that collectively, the model is statistically significant in explaining the variabilities of the target variable in the given data set. The logistic regression explanatory model suggests that _Age_ and mutated _IDH1, IDH2, PIK3R1, EGFR, GRIN2A, TP3,_ and _NF1_ genes are statistically significant and independently explain the variabilities in the GBM classification. In the ANOVA analysis, mutations in _IDH1_ and _IDH2_ genes have the highest residual deviances among the variables, suggesting that they are the most influential variables to the variances in the target variable. Variables that are positively associated with the presence of GBM include _Age_, _PIK3R1_, _GRIN2A_, and _TP53_. On the other hand, variables that are negatively associated with the presence of GBM are _IDH1, IDH2, EGFR_, and _NF1_. The coefficients and p-values are displayed in the table below.

| **Variables** | **Coefficients** | **p-value** |
| --- | --- | --- |
| IDH1 | -4.12196 | 0.000 |
| IDH2 | -3.59991 | 0.002 |
| Age | 0.03850 | 0.001 |
| PIK3R1 | 1.81795 | 0.010 |
| EGFR | -0.73849 | 0.032 |
| GRIN2A | 2.78971 | 0.005 |
| TP53 | 0.84230 | 0.036 |
| NF1 | -1.14484 | 0.005 |

_Figure 7. Logistic Regression Coefficients and P-Values_

**Coefficients Interpretation**

The coefficients are log of the chances of individuals with these characteristics having GBM compared to the reference group. Therefore, the analysis will use exponents of these coefficients to determine the mean chance of individuals with these traits to have GBM instead of LGG.

- IDH1: individuals with IDH1 mutation will have 98.4% chance of having LGG instead of GBM compared to individuals without this genetic mutation.
- IDH2: individuals with IDH2 mutation will have 97.3% chance of having LGG instead of GBM compared to individuals without this genetic mutation.
- PIK3R1: individuals with PIK3R1 mutation will have 615.9% chance of having GBM instead of LGG compared to individuals without this genetic mutation.
- EGFR: individuals with IDH2 mutation will have 52.2% chance of having LGG instead of GBM compared to individuals without this genetic mutation.
- GRIN2A: individuals with GRIN2A mutation will have 1627.6% chance of having GBM instead of LGG compared to individuals without this genetic mutation.
- TP53: individuals with TP53 mutation will have 232.2% chance of having GBM instead of LGG compared to individuals without this genetic mutation.
- NF1: individuals with NF1 mutation will have 68.2% chance of having LGG instead of GBM compared to individuals without this genetic mutation.
- Age: for every additional year of age, the chance of getting GBM instead of LGG increases by 103.9%.


## **DISCUSSION**

**Age**

About 90% of GBM are primary tumors. The Cancer Genome Atlas Network used molecular factors in addition to histology to characterize several tumor entities, making GBM one of the first tumors it analyzed (Thieme, 2020). The development of many disorders, particularly those that impact the central nervous system, is greatly influenced by advanced age. GBM are known to be very aggressive tumors that tend to affect older patients more frequently and develop without any prior disease. In a study "Glioblastoma as an age-related neurological disorder in adults" shows a median age of diagnosis at 68-70 years, while growing older is linked to a worse prognosis for these patients. In addition, advanced age of \> 65 years is connected with a high risk of disease occurrence (Kim et.al, 2019).

**Genetic Mutation**

IDH1 and IDH2, EGFR, TP53, NF1, GRIN2A, PIK3R1: The Isocitrate Dehydrogenase 1/2 are known as prognostic biomarkers. According to George Thieme in "_Glioblastoma Multiforme and Genetic Mutations: The Issue Is Not Over Yet. An Overview of the Current Literature_" reported "The 2016 WHO classification divides gliomas into LGGs and GBMs per histology.", LGGs are further divided into isocitrate dehydrogenase (IDH) 1/2 wild-type or mutant and further classified if there is a complete deletion of both the short arm of chromosome 1 and the long arm of chromosome 19 (1p/19q codeletion) (oligodendroglioma) or intact 1p/19q loci (diffuse astrocytoma)". IDH wild-type (primary or de novo GBMs) and IDH mutant (secondary GBMs) are the two types of GBMs (Thieme, 2020).

The most frequent genetic alterations in primary GBMs are loss of heterozygosity (LOH) at 10q (65% of cases), amplification or mutation of the EGFR (22-40%), the PDGFR (7%), the tumor protein 53 (TP53) mutation (28-31%), IDH1/2 mutation (5%) respectively (Thieme, 2020). In secondary GBMs, the most frequent genetic alterations observed are TP53 mutation (65% of cases), LOH at 22q (70–80%), LOH 19q (40–50%), IDH1/2 mutation (45–50%), neurofibromatosis type 1 (NF1) deletion or mutations (11%), and EGFR (5–7%). (Thieme, 2020). Inferring that TP53 mutation is a distinct and characteristic event in secondary GBM oncology and that TP53 mutation in primary GBM is possibly a result of extensive genomic instability, TP53 mutation is mostly associated with secondary GBMs. Patients with IDH mutation were seen to show a better prognosis in the various grades of glioma as well as different characteristics relative to patients with glioma with wild-type IDH (Thieme, 2020).NF1 Is known to have an influence on the tumor environment and NF1 patients, however, are also more likely to develop higher-grade infiltrating gliomas (Wood et.al., 2018)

The GRIN2A gene is found to be linked to human disease and found in neurological disorders (Vincent et. Al, 2018). It is also present in the cerebellar and lung foci suggesting its role in lung metastasis (Georgescu, 2020).

The PIK3R1 gene may serve as a therapeutic target since it affects the proliferation and motility of tumor cells in GBM (Weber et.al, 2011).

**Managerial Implications**

In conclusion, age and all the genes above play a significant role in the genetic mutation for GBM and this supports and shows they are statistically significant from the analysis that was conducted. The prognosis for patients with GBM, a very aggressive and deadly brain tumor, can be improved by early detection and treatment. Several significant managerial ramifications include:

1. Enhance Screening and Diagnostic Procedures: To identify GBM in its earliest stages, healthcare organizations should make investments in cutting-edge screening and diagnostic technologies. This necessitates allocating funds for research and development as well as educating medical professionals on how to recognize early symptoms and indicators of GBM. Tests and procedures include Imaging tests e.g. Neurologic Exam, Biopsy, Magnetic Resonance Imaging (MRI), Computed Tomography (CT), and positron emission tomography (PET).

2. Educate Medical Personnel: To keep medical personnel up to date on the most recent developments in the treatment of GBM, ongoing medical education and training programs should be implemented. The Healthcare Administration Department should make sure that their personnel are knowledgeable about the most recent standards and procedures for the early detection and treatment of GBM. E.g., Tumor grading, radiotherapy and chemotherapy, tumor treating fields (TFF) therapy, surgery, supportive care etc.

3. Revamp Treatment Pathways: To reduce delays and provide prompt access to the right interventions for GBM patients, healthcare organizations must revamp treatment pathways. Multi-specialist tumor boards may need to be established for medical professionals to evaluate each case and suggest the best course of action.

4. Monitoring and Evaluating Patient Outcomes: After an early diagnosis and course of GBM treatment, healthcare organizations should set up reliable procedures to monitor and assess patient outcomes. This data-driven strategy/analysis can aid in pinpointing areas that require improvement, evaluating the efficacy of various treatment modalities, and allocating resources as efficiently as possible. The current standard for GBM treatment is known to have been effective which has resulted in more individuals living two to four years longer (Weingart, 2013).

5. Encourage Research and Innovation: It's critical to promote innovation and research in the realm of GBM diagnosis and treatment. The exploration of novel diagnostic approaches, therapeutic modalities, and prospective biomarkers that could help early identification and individualized therapies should be supported and funded by the government and healthcare system.

**Potential Roadblocks and Limitations**

In the process of conducting the analysis, several potential factors could arise and impede the project's progress. These factors could be but not limited to:

- Data integrity: the data being used by the project is secondary data, which is susceptible to concerns such as selection bias, missing data, generalizability of results, and inaccuracy.
- Ambiguous or contradictory findings: there exists a possibility that the analysis results may contradict existing literature and findings, which could lead to more difficult interpretations of the results, and further clouding the literature body on GBM.
- Robustness: if the statistical approach of using logistic regression does not yield the necessary robustness, the project will switch to the machine learning approach with boosting methods.

## **REFERENCES**

Aly, A., Singh, P., Korytowsky, B., Ling, Y. L., Kale, H. P., Dastani, H. B., Botteman, M. F., & Norden, A. D. (2020). Survival, costs, and health care resource use by line of therapy in US Medicare patients with newly diagnosed glioblastoma: a retrospective observational study. _Neuro-oncology practice_, _7_(2), 164–175. [https://doi.org/10.1093/nop/npz042](https://doi.org/10.1093/nop/npz042)

Bielecka, J., & Markiewicz-Żukowska, R. (2020). The Influence of Nutritional and Lifestyle Factors on Glioma Incidence. Nutrients, 12(6), 1812. [https://doi.org/10.3390/nu12061812](https://doi.org/10.3390/nu12061812)

Boele, F. W., Meads, D., Jansen, F., Verdonck-de Leeuw, I. M., Heimans, J. J., Reijneveld, J. C., Short, S. C., & Klein, M. (2020). Healthcare utilization and productivity loss in glioma patients and family caregivers: the impact of treatable psychological symptoms. _Journal of neuro-oncology_, _147_(2), 485–494. [https://doi.org/10.1007/s11060-020-03454-3](https://doi.org/10.1007/s11060-020-03454-3)

Disney-Hogg, L., Sud, A., Law, P. J., Cornish, A. J., Kinnersley, B., Ostrom, Q. T., Labreche, K., Eckel-Passow, J. E., Armstrong, G. N., Claus, E. B., Il'yasova, D., Schildkraut, J., Barnholtz-Sloan, J. S., Olson, S. H., Bernstein, J. L., Lai, R. K., Swerdlow, A. J., Simon, M., Hoffmann, P., Nöthen, M. M., … Houlston, R. S. (2018). Influence of obesity-related risk factors in the aetiology of glioma. British journal of cancer, 118(7), 1020–1027. [https://doi.org/10.1038/s41416-018-0009-x](https://doi.org/10.1038/s41416-018-0009-x)

Feng, Y., Wang, J., Tan, D., Cheng, P., & Wu, A. (2019). Relationship between circulating inflammatory factors and glioma risk and prognosis: A meta-analysis. Cancer medicine, 8(17), 7454–7468. [https://doi.org/10.1002/cam4.2585](https://doi.org/10.1002/cam4.2585)

Georgescu, M.-M., & Olar, A. (2020). Genetic and histologic spatiotemporal evolution of recurrent, multifocal, multicentric and metastatic glioblastoma. Acta Neuropathologica Communications, 8(1).[https://doi.org/10.1186/s40478-020-0889-x](https://doi.org/10.1186/s40478-020-0889-x)

Glioblastoma Multiforme (GBM): Advancing Treatment for a Dangerous Brain Tumor. (2013).[https://www.hopkinsmedicine.org/health/conditions-and-diseases/glioblastoma-multiforme-gbm-advancing-treatment-for-a-dangerous-brain-tumor](https://www.hopkinsmedicine.org/health/conditions-and-diseases/glioblastoma-multiforme-gbm-advancing-treatment-for-a-dangerous-brain-tumor)

Glioblastoma Multiforme – Symptoms, Diagnosis and Treatment Options. (n.d.). Www.aans.org.[https://www.aans.org/en/Patients/Neurosurgical-Conditions-and-Treatments/Glioblastoma-Multiforme#:~:text=A%20Neurosurgeon%20Explains%3A%20Glioblastoma%20Multiforme&text=Glioblastoma%20(GBM)%2C%20also%20referred](https://www.aans.org/en/Patients/Neurosurgical-Conditions-and-Treatments/Glioblastoma-Multiforme#:~:text=A%20Neurosurgeon%20Explains%3A%20Glioblastoma%20Multiforme&text=Glioblastoma%20(GBM)%2C%20also%20referred)

Grochans, S., Cybulska, A. M., Simińska, D., Korbecki, J., Kojder, K., Chlubek, D., & Baranowska-Bosiacka, I. (2022). Epidemiology of Glioblastoma Multiforme-Literature Review. Cancers, 14(10), 2412. [https://doi.org/10.3390/cancers14102412](https://doi.org/10.3390/cancers14102412)

Jiang, S., Hill, K., Patel, D., Waldeck, A. R., Botteman, M., Aly, A., & Norden, A. D. (2017). Direct medical costs of treatment in newly-diagnosed high-grade glioma among commercially insured US patients. _Journal of medical economics_, _20_(12), 1237–1243. [https://doi.org/10.1080/13696998.2017.1364258](https://doi.org/10.1080/13696998.2017.1364258)

Kim, M., Ladomersky, E., Mozny, A., Kocherginsky, M., O'Shea, K., Reinstein, Z. Z., Zhai, L., Bell, A., Lauing, K. L., Bollu, L., Rabin, E., Dixit, K., Kumthekar, P., Platanias, L. C., Hou, L., Zheng, Y., Wu, J., Zhang, B., Hrachova, M., & Merrill, S. A. (2021). Glioblastoma as an age-related neurological disorder in adults. Neuro-Oncology Advances, 3(1).[https://doi.org/10.1093/noajnl/vdab125](https://doi.org/10.1093/noajnl/vdab125)

Lin, Z., Yang, R., Li, K., Yi, G., Li, Z., Guo, J., Zhang, Z., Junxiang, P., Liu, Y., Qi, S., & Huang, G. (2020). Establishment of age group classification for risk stratification in glioma patients. BMC neurology, 20(1), 310. [https://doi.org/10.1186/s12883-020-01888-w](https://doi.org/10.1186/s12883-020-01888-w)

Mayo Clinic. (2019). Glioblastoma - Overview - Mayo Clinic. Mayoclinic.org.[https://www.mayoclinic.org/diseases-conditions/glioblastoma/cdc-20350148](https://www.mayoclinic.org/diseases-conditions/glioblastoma/cdc-20350148)

Molinaro, A. M., Taylor, J. W., Wiencke, J. K., & Wrensch, M. R. (2019). Genetic and molecular epidemiology of adult diffuse glioma. Nature reviews. Neurology, 15(7), 405–417. [https://doi.org/10.1038/s41582-019-0220-2](https://doi.org/10.1038/s41582-019-0220-2)

Montemurro, N. (2019). Glioblastoma Multiforme and Genetic Mutations: The Issue Is Not Over Yet. An Overview of the Current Literature. Journal of Neurological Surgery Part A: Central European Neurosurgery.[https://doi.org/10.1055/s-0039-1688911](https://doi.org/10.1055/s-0039-1688911)

Myers, S. J., Yuan, H., Kang, J.-Q., Tan, F. C. K., Traynelis, S. F., & Low, C.-M. (2019). Distinct roles of GRIN2A and GRIN2B variants in neurological conditions. F1000Research, 8, 1940.[https://doi.org/10.12688/f1000research.18949.1](https://doi.org/10.12688/f1000research.18949.1)

Panje, C. M., Putora, P. M., Hundsberger, T., Hottinger, A. F., Roelcke, U., Pesce, G., Herrmann, E., & Matter-Walstra, K. (2019). Impact of treatment decision algorithms on treatment costs in recurrent glioblastoma: a health economic study. Swiss medical weekly, 149, w20153. [https://doi.org/10.4414/smw.2019.20153](https://doi.org/10.4414/smw.2019.20153)

Ray, S., Bonafede, M. M., & Mohile, N. A. (2014). Treatment Patterns, Survival, and Healthcare Costs of Patients with Malignant Gliomas in a Large US Commercially Insured Population. American health & drug benefits, 7(3), 140–149.

Sato, K., Tahata, K., & Akimoto, K. (2020). Five Genes Associated With Survival in Patients With Lower-grade Gliomas Were Identified by Information-theoretical Analysis. _Anticancer research_, _40_(5), 2777–2785. [https://doi.org/10.21873/anticanres.14250](https://doi.org/10.21873/anticanres.14250)

Shergalis, A., Bankhead, A., 3rd, Luesakul, U., Muangsin, N., & Neamati, N. (2018). Current Challenges and Opportunities in Treating Glioblastoma. _Pharmacological reviews_, _70_(3), 412–445. [https://doi.org/10.1124/pr.117.014944](https://doi.org/10.1124/pr.117.014944)

Tasci,Erdal, Camphausen,Kevin, Krauze,Andra Valentina, and Zhuge,Ying. (2022). Glioma Grading Clinical and Mutation Features Dataset. UCI Machine Learning Repository. [https://doi.org/10.24432/C5R62J](https://doi.org/10.24432/C5R62J).

Thakkar, J. P., Dolecek, T. A., Horbinski, C., Ostrom, Q. T., Lightner, D. D., Barnholtz-Sloan, J. S., & Villano, J. L. (2014). Epidemiologic and molecular prognostic review of glioblastoma. Cancer epidemiology, biomarkers & prevention : a publication of the American Association for Cancer Research, cosponsored by the American Society of Preventive Oncology, 23(10), 1985–1996. [https://doi.org/10.1158/1055-9965.EPI-14-0275](https://doi.org/10.1158/1055-9965.EPI-14-0275)

Tuohy, K., Ba, D. M., Bhanja, D., Leslie, D., Liu, G., & Mansouri, A. (2023). Early costs and complications of first-line low-grade glioma treatment using a large national database: Limitations and future perspectives. _Frontiers in surgery_, _10_, 1001741. [https://doi.org/10.3389/fsurg.2023.1001741](https://doi.org/10.3389/fsurg.2023.1001741)

Weber, G. L., Parat, M.-O., Binder, Z. A., Gallia, G. L., & Riggins, G. J. (2011). Abrogation of PIK3CA or PIK3R1 reduces proliferation, migration, and invasion in glioblastoma multiforme cells. Oncotarget, 2(11), 833–849. [https://doi.org/10.18632/oncotarget.346](https://doi.org/10.18632/oncotarget.346)

Wiedmann, M. K. H., Brunborg, C., Di Ieva, A., Lindemann, K., Johannesen, T. B., Vatten, L., Helseth, E., & Zwart, J. A. (2017). The impact of body mass index and height on the risk for

glioblastoma and other glioma subgroups: a large prospective cohort study. Neuro-oncology, 19(7), 976–985. [https://doi.org/10.1093/neuonc/now272](https://doi.org/10.1093/neuonc/now272)

Wood, M. D., Mukherjee, J., & Pieper, R. O. (2018). Neurofibromin knockdown in glioma cell lines is associated with changes in cytokine and chemokine secretion in vitro. Scientific Reports, 8(1).https://doi.org/10.1038/s41598-018-24046-2


## **APPENDIX**

![](RackMultipart20231107-1-4mtcmt_html_68987f401f9bb3.png)

_Figure 1. Analysis Work Flow_

![](RackMultipart20231107-1-4mtcmt_html_19adf93b7a897e36.png)

_Figure 2. Distribution of GBM Classes_

![](RackMultipart20231107-1-4mtcmt_html_aeffab36374811af.png)

_Figure 3. Correlation Matrix Between Features_

![](RackMultipart20231107-1-4mtcmt_html_30d2da3b5a056b94.png)

_Figure 4. Boruta Feature Importance_

![](RackMultipart20231107-1-4mtcmt_html_c5f9c36fa2173697.png)

_Figure 5. RFE Cross Validation Accuracy_

| **Classification Algorithm** | **Accuracy** |
| --- | --- |
| Logistic Regression | 0.8968254 |
| Linear SVM | 0.8888889 |
| Random Forest | 0.8849206 |
| AdaBoost | 0.8769841 |
| Non-linear SVM | 0.8690476 |
| Naïve Bayes | 0.7301587 |

_Figure 6. Accuracy Comparison Between Classification Algorithms_
