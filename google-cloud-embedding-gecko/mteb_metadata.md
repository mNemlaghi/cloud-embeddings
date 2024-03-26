---
tags:
- mteb
model-index:
- name: textembedding-gecko@003
  results:
  - task:
      type: STS
    dataset:
      type: mteb/biosses-sts
      name: MTEB BIOSSES
      config: default
      split: test
      revision: d3fb88f8f02e40887cd149695127462bbcf29b4a
    metrics:
    - type: cos_sim_pearson
      value: 88.55426978385223
    - type: cos_sim_spearman
      value: 85.79231628190917
    - type: euclidean_pearson
      value: 87.41970677659403
    - type: euclidean_spearman
      value: 85.79231628190917
    - type: manhattan_pearson
      value: 87.42033604166257
    - type: manhattan_spearman
      value: 85.99051356905314
  - task:
      type: STS
    dataset:
      type: mteb/sts12-sts
      name: MTEB STS12
      config: default
      split: test
      revision: a0d554a64d88156834ff5ae9920b964011b16384
    metrics:
    - type: cos_sim_pearson
      value: 75.85717778248366
    - type: cos_sim_spearman
      value: 69.621162638098
    - type: euclidean_pearson
      value: 72.2712537394482
    - type: euclidean_spearman
      value: 69.62114220687268
    - type: manhattan_pearson
      value: 72.27083630503148
    - type: manhattan_spearman
      value: 69.6463031406694
  - task:
      type: STS
    dataset:
      type: mteb/sts13-sts
      name: MTEB STS13
      config: default
      split: test
      revision: 7e90230a92c190f1bf69ae9002b8cea547a64cca
    metrics:
    - type: cos_sim_pearson
      value: 85.18536547294273
    - type: cos_sim_spearman
      value: 85.4633222493931
    - type: euclidean_pearson
      value: 84.97649497806233
    - type: euclidean_spearman
      value: 85.4633222493931
    - type: manhattan_pearson
      value: 84.93277675916282
    - type: manhattan_spearman
      value: 85.40687608068971
  - task:
      type: STS
    dataset:
      type: mteb/sts14-sts
      name: MTEB STS14
      config: default
      split: test
      revision: 6031580fec1f6af667f0bd2da0a551cf4f0b2375
    metrics:
    - type: cos_sim_pearson
      value: 81.19388058502382
    - type: cos_sim_spearman
      value: 77.78783608354006
    - type: euclidean_pearson
      value: 80.31374353492374
    - type: euclidean_spearman
      value: 77.78778819897273
    - type: manhattan_pearson
      value: 80.2734419297265
    - type: manhattan_spearman
      value: 77.7475302399356
  - task:
      type: STS
    dataset:
      type: mteb/sts15-sts
      name: MTEB STS15
      config: default
      split: test
      revision: ae752c7c21bf194d8b67fd573edf7ae58183cbe3
    metrics:
    - type: cos_sim_pearson
      value: 84.69477613621183
    - type: cos_sim_spearman
      value: 85.9216860882149
    - type: euclidean_pearson
      value: 85.52305929161254
    - type: euclidean_spearman
      value: 85.92165459234619
    - type: manhattan_pearson
      value: 85.49135438248668
    - type: manhattan_spearman
      value: 85.88921057788731
  - task:
      type: STS
    dataset:
      type: mteb/sts16-sts
      name: MTEB STS16
      config: default
      split: test
      revision: 4d8694f8f0e0100860b497b999b3dbed754a0513
    metrics:
    - type: cos_sim_pearson
      value: 82.99622076915341
    - type: cos_sim_spearman
      value: 84.27170496574477
    - type: euclidean_pearson
      value: 83.93603562294123
    - type: euclidean_spearman
      value: 84.27170496574477
    - type: manhattan_pearson
      value: 83.9154125146616
    - type: manhattan_spearman
      value: 84.25467348177617
  - task:
      type: STS
    dataset:
      type: mteb/sts17-crosslingual-sts
      name: MTEB STS17 (en-en)
      config: en-en
      split: test
      revision: af5e6fb845001ecf41f4c1e033ce921939a2a68d
    metrics:
    - type: cos_sim_pearson
      value: 89.24669774812011
    - type: cos_sim_spearman
      value: 89.96063236506942
    - type: euclidean_pearson
      value: 90.32174162104575
    - type: euclidean_spearman
      value: 89.96063236506942
    - type: manhattan_pearson
      value: 90.33271196047316
    - type: manhattan_spearman
      value: 90.0432007442401
  - task:
      type: STS
    dataset:
      type: mteb/stsbenchmark-sts
      name: MTEB STSBenchmark
      config: default
      split: test
      revision: b0fddb56ed78048fa8b90373c8a3cfc37b684831
    metrics:
    - type: cos_sim_pearson
      value: 83.18490942903185
    - type: cos_sim_spearman
      value: 83.36727866698648
    - type: euclidean_pearson
      value: 83.73749276887087
    - type: euclidean_spearman
      value: 83.36727866698648
    - type: manhattan_pearson
      value: 83.70507180312714
    - type: manhattan_spearman
      value: 83.33412995257402
---