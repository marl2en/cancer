# cancer
Analysis by gender, age group, cancer type and prediction 


# Analysis of Cancer Data in Sweden 1970-2021. Prediction for 2022. 

The data is presented in 5-years age groups AG = ['0_4', '5_9','10_14','15_19','20_24','25_29','30_34','35_39','40_44','45_49','50_54','55_59','60_64','65_69','70_74','75_79','80_84','85+'] 

cancer types are:
 ['breast','other_skin_cancer','uterus_cervix_corpus','colon_rectum','urinary_tract','melanoma','lung_trachea_bronchi','brain','esophagus','kidney', 'liver_bile_duct', 'other_endocrine_glandes', 
     'Hodgkin_lymphoma','Multiple_myeloma', 'Non-Hodgkin_lymphoma','ovarian', 'pancreas', 'stomach', 'thyroid', 'unspecified_localization','prostata','testicle']

Abbreviations:
- other_skin_cancer: OSC,
- uterus_cervix_corpus: uterus,
- colon_rectum: colon,
- urinary_tract: UT,
- lung_trachea_bronchi: lung,
- liver_bile_duct: liver,
- other_endocrine_glandes: OEG
- Hodgkin_lymphoma: HL,
- Multiple_myeloma: MM, 
- Non-Hodgkin_lymphoma: NHL,
- unspecified_localization: UL


The data is downloaded from Socialstyrelsen the National Board of Health in Sweden (https://www.socialstyrelsen.se/en/ )

Then the data is parsed and arrange in a json file according to:

numbers[gender]['age'][AgeGroup]['cancer_types'][type][parameter]

Parameters are:
- real: real count of cancer cases in this age group
- ssm_mean/lower/upper: State Space Model forecast
- count_mean/lower/upper: count forecast with pymc3 model
- incidence_mean/lower/upper: incidence forecast with pymc3 model

see numbers_pymc3.json

Population size per age group is in file pop_ag.csv 

Incidence is given as new cancer cases per 100,000 in respective age group 


https://austinrochford.com/posts/apc-pymc.html


# prediction of incidence for 2023



lower: lower confidence interval (-2 SD)
upper: upper confidence interval (+2 SD)



|Value|CT|Gender|20_24|25_29|30_34|35_39|40_44|45_49|50_54|55_59|60_64|65_69|70_74|75_79|80_84|85+|
|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
|lower|breast|Men|0|0|0|0|0|0|0|0|0|0|0|0|0|0|
|mean|breast|Men|0|0|0|0|0|0|0|1|1|2|4|4|5|6|
|upper|breast|Men|0|0|0|0|0|0|1|3|4|7|10|10|11|13|
|lower|breast|Women|0|2|17|49|107|178|193|208|264|304|346|309|322|285|
|mean|breast|Women|0|10|36|87|180|290|310|338|419|490|545|490|520|458|
|upper|breast|Women|2|18|55|125|253|402|428|468|574|677|744|671|718|632|
|lower|OSC|Men|0|0|0|0|0|1|5|14|37|84|192|345|569|889|
|mean|OSC|Men|0|0|0|1|3|7|15|33|76|158|339|611|994|1574|
|upper|OSC|Men|0|1|2|4|7|13|25|52|115|232|485|878|1418|2258|
|lower|OSC|Women|0|0|0|0|0|2|8|17|33|60|111|196|282|474|
|mean|OSC|Women|0|0|1|2|5|12|23|40|69|120|207|359|533|873|
|upper|OSC|Women|0|1|3|6|11|21|38|62|106|180|304|523|783|1271|
|lower|uterus|Women|0|2|7|9|12|15|22|28|36|47|56|61|68|59|
|mean|uterus|Women|2|10|19|23|26|32|43|55|67|84|101|109|119|102|
|upper|uterus|Women|5|18|31|37|40|49|64|82|99|121|145|157|170|146|
|lower|colon|Men|0|0|0|1|4|11|23|42|74|116|174|227|279|271|
|mean|colon|Men|0|1|3|7|14|25|46|74|126|196|283|372|452|445|
|upper|colon|Men|2|4|7|13|24|39|69|107|178|276|392|517|624|619|
|lower|colon|Women|0|0|0|1|4|11|23|36|58|89|125|168|219|207|
|mean|colon|Women|1|1|3|7|13|25|43|65|99|150|208|277|358|338|
|upper|colon|Women|4|4|7|13|22|40|64|95|140|212|290|387|497|468|
|lower|UT|Men|0|0|0|0|0|1|5|14|31|62|103|150|186|200|
|mean|UT|Men|0|0|0|1|3|7|16|31|59|110|182|259|314|339|
|upper|UT|Men|0|1|2|3|7|14|26|47|88|158|261|368|443|479|
|lower|UT|Women|0|0|0|0|0|0|0|3|9|15|27|36|44|40|
|mean|UT|Women|0|0|0|0|1|2|5|12|23|36|55|72|84|79|
|upper|UT|Women|0|0|0|1|3|6|11|20|38|56|83|108|125|119|
|lower|melanoma|Men|0|0|1|4|8|15|23|33|44|58|74|97|116|136|
|mean|melanoma|Men|2|4|8|14|23|35|49|66|85|109|141|181|215|245|
|upper|melanoma|Men|5|9|15|24|37|54|74|98|127|159|208|266|314|354|
|lower|melanoma|Women|0|2|5|9|17|24|30|30|34|40|51|59|61|64|
|mean|melanoma|Women|5|9|15|24|37|51|60|62|68|78|96|112|115|121|
|upper|melanoma|Women|10|17|26|39|57|77|90|93|102|115|142|164|169|179|
|lower|lung|Men|0|0|0|0|0|0|4|12|28|60|97|127|123|86|
|mean|lung|Men|0|0|0|1|2|6|12|26|55|110|171|221|217|153|
|upper|lung|Men|0|0|1|3|6|11|21|41|83|160|245|315|311|220|
|lower|lung|Women|0|0|0|0|0|1|5|16|38|70|109|120|101|57|
|mean|lung|Women|0|0|0|1|3|8|17|35|76|133|197|225|186|108|
|upper|lung|Women|0|0|1|3|8|15|28|54|114|195|286|330|271|159|
|lower|brain|Men|0|0|0|1|2|4|6|7|10|11|12|12|8|6|
|mean|brain|Men|3|5|6|8|10|14|18|21|27|29|30|29|23|18|
|upper|brain|Men|8|10|12|15|18|24|31|34|43|47|48|46|37|30|
|lower|brain|Women|0|0|0|1|3|6|8|9|11|13|14|12|7|4|
|mean|brain|Women|3|4|6|8|12|18|22|25|29|31|33|28|21|14|
|upper|brain|Women|6|8|12|15|21|30|36|40|46|49|52|45|35|25|
|lower|esophagus|Men|0|0|0|0|0|0|0|0|3|7|12|14|14|13|
|mean|esophagus|Men|0|0|0|0|0|1|3|7|14|22|30|34|35|32|
|upper|esophagus|Men|0|0|0|0|0|3|7|14|24|37|49|53|55|50|
|lower|esophagus|Women|0|0|0|0|0|0|0|0|0|0|0|2|2|2|
|mean|esophagus|Women|0|0|0|0|0|0|0|2|3|6|9|12|12|12|
|upper|esophagus|Women|0|0|0|0|0|0|2|5|8|13|18|22|22|22|
|lower|kidney|Men|0|0|0|0|0|3|7|10|17|23|32|38|32|21|
|mean|kidney|Men|0|0|0|2|6|12|20|27|40|51|67|77|68|46|
|upper|kidney|Men|0|1|2|6|13|21|34|44|62|79|103|117|104|72|
|lower|kidney|Women|0|0|0|0|0|0|1|3|5|9|13|16|14|8|
|mean|kidney|Women|0|0|0|0|2|4|8|13|18|24|33|38|36|25|
|upper|kidney|Women|0|0|1|2|5|9|16|24|30|40|53|60|58|41|
|lower|liver|Men|0|0|0|0|0|0|1|6|13|20|27|28|28|22|
|mean|liver|Men|0|0|0|0|2|4|9|18|33|49|60|65|64|51|
|upper|liver|Men|0|0|1|2|5|9|17|31|53|78|94|101|100|79|
|lower|liver|Women|0|0|0|0|0|0|0|1|4|8|14|18|21|17|
|mean|liver|Women|0|0|0|0|0|2|4|8|15|23|35|44|49|40|
|upper|liver|Women|0|0|0|1|3|5|9|16|25|39|56|70|78|63|
|lower|OEG|Men|0|0|0|0|0|0|2|2|4|6|6|6|4|2|
|mean|OEG|Men|1|2|4|5|6|7|11|13|15|19|21|22|16|10|
|upper|OEG|Men|4|6|9|10|13|15|20|24|27|31|35|37|29|17|
|lower|OEG|Women|0|1|1|2|3|4|7|7|8|8|9|8|5|0|
|mean|OEG|Women|6|9|9|11|14|15|21|22|22|23|25|24|16|7|
|upper|OEG|Women|12|16|17|21|24|27|35|36|37|39|41|39|28|14|
|lower|HL|Men|0|0|0|0|0|0|0|0|0|0|0|0|0|0|
|mean|HL|Men|3|2|2|2|2|1|1|1|2|2|3|2|2|2|
|upper|HL|Men|7|6|6|5|5|4|4|4|5|6|7|6|6|5|
|lower|HL|Women|0|0|0|0|0|0|0|0|0|0|0|0|0|0|
|mean|HL|Women|3|2|2|1|1|1|0|0|1|1|1|1|1|1|
|upper|HL|Women|8|6|5|4|3|3|3|3|3|4|4|4|4|3|
|lower|MM|Men|0|0|0|0|0|0|0|1|5|8|13|18|21|18|
|mean|MM|Men|0|0|0|0|1|2|5|9|16|23|32|42|48|41|
|upper|MM|Men|0|0|0|1|3|5|10|16|26|37|52|67|75|64|
|lower|MM|Women|0|0|0|0|0|0|0|0|1|4|7|10|12|7|
|mean|MM|Women|0|0|0|0|0|1|3|6|9|15|21|26|29|22|
|upper|MM|Women|0|0|0|0|2|4|7|12|17|26|35|43|47|36|
|lower|NHL|Men|0|0|0|0|0|2|4|8|14|24|33|47|58|52|
|mean|NHL|Men|1|1|2|3|6|10|15|23|33|50|70|95|112|106|
|upper|NHL|Men|3|3|5|8|12|18|26|37|53|77|108|144|166|160|
|lower|NHL|Women|0|0|0|0|0|0|2|5|9|13|20|28|32|28|
|mean|NHL|Women|0|1|1|2|3|6|10|16|23|34|46|61|68|60|
|upper|NHL|Women|2|3|4|6|8|13|19|27|38|55|72|94|103|93|
|lower|ovarian|Women|0|0|0|0|0|3|5|8|11|14|17|18|15|9|
|mean|ovarian|Women|1|1|2|3|6|11|16|21|26|32|37|38|33|22|
|upper|ovarian|Women|3|4|5|7|12|19|27|33|42|49|57|57|51|35|
|lower|pancreas|Men|0|0|0|0|0|0|0|3|9|17|27|33|30|22|
|mean|pancreas|Men|0|0|0|0|0|2|7|13|24|40|59|71|66|50|
|upper|pancreas|Men|0|0|0|1|2|6|13|24|40|63|91|109|101|78|
|lower|pancreas|Women|0|0|0|0|0|0|0|2|7|15|23|31|29|21|
|mean|pancreas|Women|0|0|0|0|0|2|6|11|22|36|52|66|64|49|
|upper|pancreas|Women|0|0|0|0|2|6|12|20|37|56|80|101|98|76|
|lower|stomach|Men|0|0|0|0|0|0|0|2|5|10|16|22|26|28|
|mean|stomach|Men|0|0|0|0|1|3|5|9|16|25|36|48|56|59|
|upper|stomach|Men|0|0|1|2|5|6|10|17|27|41|56|74|86|90|
|lower|stomach|Women|0|0|0|0|0|0|0|0|1|3|6|9|11|11|
|mean|stomach|Women|0|0|0|0|1|2|3|5|8|12|18|25|29|28|
|upper|stomach|Women|0|0|1|2|4|6|7|10|16|21|30|40|48|46|
|lower|thyroid|Men|0|0|0|0|0|0|0|0|0|0|1|1|1|0|
|mean|thyroid|Men|0|1|3|4|5|6|6|7|8|9|12|11|10|7|
|upper|thyroid|Men|2|4|7|9|11|13|13|14|16|18|22|21|19|15|
|lower|thyroid|Women|0|1|3|4|6|6|6|4|4|4|5|4|4|3|
|mean|thyroid|Women|5|9|15|17|20|21|19|17|16|16|18|17|16|13|
|upper|thyroid|Women|11|17|26|31|34|35|32|29|27|28|32|29|28|23|
|lower|UL|Men|0|0|0|0|0|0|0|0|1|4|7|10|14|15|
|mean|UL|Men|0|0|0|0|0|1|3|6|9|14|21|27|35|35|
|upper|UL|Men|0|0|0|1|2|4|7|12|17|25|36|45|56|56|
|lower|UL|Women|0|0|0|0|0|0|0|0|3|6|10|17|22|21|
|mean|UL|Women|0|0|0|0|1|2|4|7|12|19|27|40|49|49|
|upper|UL|Women|0|0|1|1|3|5|9|14|21|32|44|62|76|76|
|lower|prostata|Men|0|0|0|0|0|8|58|145|300|415|481|437|363|238|
|mean|prostata|Men|0|0|0|0|2|23|117|284|553|771|882|821|667|439|
|upper|prostata|Men|0|0|0|0|5|39|175|423|806|1128|1284|1206|972|639|
|lower|testicle|Men|1|5|5|6|3|2|0|0|0|0|0|0|0|0|
|mean|testicle|Men|10|18|20|19|14|10|6|3|2|1|1|0|0|0|
|upper|testicle|Men|18|31|34|33|25|19|12|8|6|4|4|2|2|3|
