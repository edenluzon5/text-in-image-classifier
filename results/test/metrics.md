# Results on the test split (n = 220)

| metric | value | 95% bootstrap interval |
|---|---:|---:|
| precision | 1.000 | 1.000 – 1.000 |
| recall | 0.945 | 0.900 – 0.982 |
| f1 | 0.972 | 0.947 – 0.991 |
| accuracy | 0.973 | 0.950 – 0.991 |

Alcohol images found: 104. Missed: 6. False alarms: 0. Non-alcohol correct: 110.

## Accuracy by confidence

| confidence | images | correct | accuracy |
|---|---:|---:|---:|
| high | 88 | 88 | 1.000 |
| medium | 100 | 100 | 1.000 |
| low | 32 | 26 | 0.812 |

## Errors (6)

| file | true label | prediction | reason | what OCR read |
|---|---|---|---|---|
| 0124a.jpg | alcohol | non_alcohol | no alcohol term found | original: dpin goitibe betimilgot ubpel te / mirror: Get readu for limited edition nig / rot35: alittdpin goitibe betimilhot ubpet teo / upscale2: liptxbooeatif |
| 0209a.jpg | alcohol | non_alcohol | no alcohol term found | rot35: 一 / rot-35: 1 / upscale2: K / rot-90: TOOPTO / upscale4: K |
| 0368a.jpg | alcohol | non_alcohol | no alcohol term found | rot-35: 1 / upscale2: Uet reta yexeinw ntooma rot / rot-90: Uexlaintw ntlom / upscale4: Get rea yexainw ntooma rot |
| 0372a.jpg | alcohol | non_alcohol | no alcohol term found | mirror: merfofPcelebration / upscale2: oto / rot180: fotteepatinm / rot90: fiteplaain / upscale4: ACSW / sharpen: oito |
| 0384a.jpg | alcohol | non_alcohol | no alcohol term found | upscale2: titr / rot180: te |
| 3020a.jpg | alcohol | non_alcohol | no alcohol term found | original: refrk / mirror: berb mbouberb ueb  be d / upscale2: Get ready for discount checlevodin chack / rot180: Get ready for dcount ch chck / rot90: Gt ready  |
