# Leaf Classification — Lite Task Description

## Task description
The objective of this playground competition is to use binary leaf images and extracted features, including shape, margin & texture, to accurately identify 99 species of plants. Leaves, due to their volume, prevalence, and unique characteristics, are an effective means of differentiating plant species. They also …

## Task objective
- **Input:** Each test sample as defined by the competition `test` split and `sample_submission.csv` rows.
- **Output:** You must submit a csv file with the image id, all candidate species names, and a probability for each species.

## Target metric (evaluation)
Submissions are evaluated using the multi-class logarithmic loss. Each image has been labeled with one true species. For each image, you must submit a set of predicted probabilities (one for every species). The formula is then, \text{logloss} = -\frac{1}{N} \sum_{i=1}^N …

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** `id` plus 99 target columns exactly matching `sample_submission.csv`; do not collapse them into a generic `prediction` column.
```
id,Acer_Capillipes,Acer_Circinatum,Acer_Mono,Acer_Opalus,Acer_Palmatum,Acer_Pictum,Acer_Platanoids,Acer_Rubrum,Acer_Rufinerve,Acer_Saccharinum,Alnus_Cordata,Alnus_Maximowiczii,Alnus_Rubra,Alnus_Sieboldiana,Alnus_Viridis,Arundinaria_Simonii,Betula_Austrosinensis,Betula_Pendula,Callicarpa_Bodinieri,Castanea_Sativa,Celtis_Koraiensis,Cercis_Siliquastrum,Cornus_Chinensis,Cornus_Controversa,Cornus_Macrophylla,Cotinus_Coggygria,Crataegus_Monogyna,Cytisus_Battandieri,Eucalyptus_Glaucescens,Eucalyptus_Neglecta,Eucalyptus_Urnigera,Fagus_Sylvatica,Ginkgo_Biloba,Ilex_Aquifolium,Ilex_Cornuta,Liquidambar_Styraciflua,Liriodendron_Tulipifera,Lithocarpus_Cleistocarpus,Lithocarpus_Edulis,Magnolia_Heptapeta,Magnolia_Salicifolia,Morus_Nigra,Olea_Europaea,Phildelphus,Populus_Adenopoda,Populus_Grandidentata,Populus_Nigra,Prunus_Avium,Prunus_X_Shmittii,Pterocarya_Stenoptera,Quercus_Afares,Quercus_Agrifolia,Quercus_Alnifolia,Quercus_Brantii,Quercus_Canariensis,Quercus_Castaneifolia,Quercus_Cerris,Quercus_Chrysolepis,Quercus_Coccifera,Quercus_Coccinea,Quercus_Crassifolia,Quercus_Crassipes,Quercus_Dolicholepis,Quercus_Ellipsoidalis,Quercus_Greggii,Quercus_Hartwissiana,Quercus_Ilex,Quercus_Imbricaria,Quercus_Infectoria_sub,Quercus_Kewensis,Quercus_Nigra,Quercus_Palustris,Quercus_Phellos,Quercus_Phillyraeoides,Quercus_Pontica,Quercus_Pubescens,Quercus_Pyrenaica,Quercus_Rhysophylla,Quercus_Rubra,Quercus_Semecarpifolia,Quercus_Shumardii,Quercus_Suber,Quercus_Texana,Quercus_Trojana,Quercus_Variabilis,Quercus_Vulcanica,Quercus_x_Hispanica,Quercus_x_Turneri,Rhododendron_x_Russellianum,Salix_Fragilis,Salix_Intergra,Sorbus_Aria,Tilia_Oliveri,Tilia_Platyphyllos,Tilia_Tomentosa,Ulmus_Bergmanniana,Viburnum_Tinus,Viburnum_x_Rhytidophylloides,Zelkova_Serrata
1202,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102,0.010101010101010102
```

## Dataset and construction
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format
- **images/** - the image files (each image is named with its corresponding id)
