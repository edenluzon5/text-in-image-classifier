# Results on the val split (n = 220)

| metric | value | 95% bootstrap interval |
|---|---:|---:|
| precision | 1.000 | 1.000 – 1.000 |
| recall | 0.945 | 0.901 – 0.983 |
| f1 | 0.972 | 0.948 – 0.991 |
| accuracy | 0.973 | 0.950 – 0.991 |

Alcohol images found: 104. Missed: 6. False alarms: 0. Non-alcohol correct: 110.

## Accuracy by confidence

| confidence | images | correct | accuracy |
|---|---:|---:|---:|
| high | 71 | 71 | 1.000 |
| medium | 112 | 112 | 1.000 |
| low | 37 | 31 | 0.838 |

## Errors (6)

| file | true label | prediction | reason | what OCR read |
|---|---|---|---|---|
| 0032a.jpg | alcohol | non_alcohol | no alcohol term found | rot35: 品 / mirror_rot35: 0 / mirror_rot-35: 品 / upscale4: J lela discount 6 |
| 0085a.jpg | alcohol | non_alcohol | no alcohol term found | upscale4: C |
| 0155a.jpg | alcohol | non_alcohol | no alcohol term found | original: cimpimported / mirror: hupolfeq / rot35: imported / upscale2: nported / rot180: imported / rot90: imported / rot-90: imported / mirror_rot35: baofor / |
| 0349a.jpg | alcohol | non_alcohol | no alcohol term found | original: Get ready foR bB ow m-nisoger / mirror: TOoO hn BE W. Skd Sot yoDaT taD / upscale2: Get ready for be w m niog / rot90: a      a / rot-90: Get ready fo |
| 0375a.jpg | alcohol | non_alcohol | no alcohol term found | original: out check dliecemst badhweloer hmperted / mirror: noupnd twoiowhed tameth toodo twe / rot35: 2 / upscale2: Check out check lleceemst ibendweieer imper |
| 0397a.jpg | alcohol | non_alcohol | no alcohol term found | original: OÑE WAY NO ENTRY arty smim / mirror: ONE YAW ENTRY omima yhe g.oolt / rot35: ONE WAY NO ENTRY / rot-35: WAY / upscale2: OÑE WAY NO ENTRY / rot180: ENT |
