# Calibration report (plausibility evidence only)

Created 2026-10-10T23:52:23+00:00 - version `calib-202610102352` - 1500 sampled homes.

NEED targets are **placeholders** until replaced with the published tables. Matching cell medians shows the simulator sits in a realistic range; it does not show home-level accuracy.

## Parameters

| parameter | start | calibrated |
|---|---|---|
| kappa_global | 0.65 | 0.7028 |
| appliance_scale_global | 1.0 | 0.9280 |
| air_change_scale_global | 1.0 | 1.0154 |
| boiler_efficiency_offset_global | 0.0 | -0.0027 |

Objective (weighted mean squared log ratio + regulariser): before 0.02459, after 0.01815.

## Cell medians (kWh/yr)

| fuel | type | age | n | NEED | sim before | sim after |
|---|---|---|---|---|---|---|
| electricity | bungalow | D | 16 | 2,514 | 2,850 | 2,645 |
| electricity | bungalow | E | 27 | 2,514 | 2,822 | 2,619 |
| electricity | bungalow | F | 15 | 2,514 | 2,742 | 2,544 |
| electricity | detached | A | 16 | 3,718 | 3,541 | 3,286 |
| electricity | detached | B | 14 | 3,645 | 4,080 | 3,786 |
| electricity | detached | C | 27 | 3,645 | 4,279 | 3,971 |
| electricity | detached | D | 20 | 3,572 | 3,792 | 3,519 |
| electricity | detached | E | 25 | 3,572 | 3,798 | 3,525 |
| electricity | detached | F | 15 | 3,572 | 3,871 | 3,592 |
| electricity | detached | G | 18 | 3,645 | 4,180 | 3,879 |
| electricity | detached | I | 14 | 3,718 | 3,601 | 3,342 |
| electricity | detached | K | 14 | 3,754 | 3,280 | 3,044 |
| electricity | end_terrace | B | 13 | 2,646 | 3,149 | 2,922 |
| electricity | end_terrace | D | 12 | 2,593 | 2,855 | 2,649 |
| electricity | end_terrace | E | 15 | 2,593 | 2,744 | 2,546 |
| electricity | end_terrace | K | 12 | 2,725 | 3,029 | 2,811 |
| electricity | flat | A | 18 | 2,148 | 2,255 | 2,092 |
| electricity | flat | B | 14 | 2,106 | 2,671 | 2,479 |
| electricity | flat | C | 20 | 2,106 | 2,722 | 2,526 |
| electricity | flat | D | 19 | 2,064 | 2,565 | 2,380 |
| electricity | flat | E | 39 | 2,064 | 2,530 | 2,348 |
| electricity | flat | F | 22 | 2,064 | 2,741 | 2,544 |
| electricity | flat | G | 23 | 2,106 | 2,604 | 2,416 |
| electricity | flat | H | 13 | 2,106 | 2,717 | 2,522 |
| electricity | flat | I | 32 | 2,148 | 2,727 | 2,531 |
| electricity | flat | J | 32 | 2,148 | 2,549 | 2,366 |
| electricity | flat | K | 40 | 2,169 | 2,527 | 2,345 |
| electricity | semi | A | 38 | 2,947 | 3,339 | 3,123 |
| electricity | semi | B | 32 | 2,889 | 2,968 | 2,754 |
| electricity | semi | C | 35 | 2,889 | 2,740 | 2,543 |
| electricity | semi | D | 44 | 2,831 | 3,162 | 2,935 |
| electricity | semi | E | 35 | 2,831 | 3,189 | 2,959 |
| electricity | semi | F | 22 | 2,831 | 2,852 | 2,660 |
| electricity | semi | G | 26 | 2,889 | 2,717 | 2,521 |
| electricity | semi | H | 24 | 2,889 | 3,106 | 2,883 |
| electricity | semi | I | 18 | 2,947 | 2,747 | 2,550 |
| electricity | semi | J | 22 | 2,947 | 3,054 | 2,835 |
| electricity | semi | K | 32 | 2,976 | 2,976 | 2,762 |
| electricity | terraced | A | 73 | 2,616 | 2,880 | 2,689 |
| electricity | terraced | B | 50 | 2,565 | 2,715 | 2,519 |
| electricity | terraced | C | 25 | 2,565 | 2,698 | 2,504 |
| electricity | terraced | D | 20 | 2,514 | 2,956 | 2,744 |
| electricity | terraced | E | 27 | 2,514 | 3,054 | 2,834 |
| electricity | terraced | G | 21 | 2,565 | 2,726 | 2,530 |
| electricity | terraced | K | 12 | 2,642 | 3,552 | 3,296 |
| gas | bungalow | D | 15 | 11,371 | 12,573 | 13,411 |
| gas | bungalow | E | 25 | 10,819 | 11,445 | 12,214 |
| gas | bungalow | F | 13 | 10,488 | 9,887 | 10,531 |
| gas | detached | A | 15 | 18,239 | 20,493 | 22,134 |
| gas | detached | B | 15 | 17,192 | 21,949 | 23,596 |
| gas | detached | C | 23 | 16,445 | 15,844 | 16,940 |
| gas | detached | D | 22 | 15,398 | 19,658 | 21,085 |
| gas | detached | E | 23 | 14,651 | 15,606 | 16,771 |
| gas | detached | F | 16 | 14,202 | 16,041 | 17,157 |
| gas | detached | G | 14 | 13,455 | 14,965 | 16,057 |
| gas | detached | I | 14 | 12,558 | 10,373 | 10,969 |
| gas | detached | K | 13 | 11,512 | 10,397 | 11,034 |
| gas | end_terrace | D | 12 | 11,371 | 10,109 | 10,870 |
| gas | end_terrace | E | 13 | 10,819 | 9,748 | 10,452 |
| gas | end_terrace | K | 14 | 8,501 | 5,949 | 6,222 |
| gas | flat | A | 19 | 9,120 | 7,678 | 8,151 |
| gas | flat | B | 15 | 8,596 | 8,779 | 9,417 |
| gas | flat | C | 19 | 8,222 | 6,297 | 6,633 |
| gas | flat | D | 18 | 7,699 | 6,691 | 7,162 |
| gas | flat | E | 40 | 7,326 | 6,444 | 6,811 |
| gas | flat | F | 19 | 7,101 | 6,676 | 6,985 |
| gas | flat | G | 22 | 6,728 | 6,192 | 6,519 |
| gas | flat | H | 12 | 6,503 | 5,792 | 6,115 |
| gas | flat | I | 30 | 6,279 | 4,677 | 4,902 |
| gas | flat | J | 30 | 5,980 | 4,824 | 5,038 |
| gas | flat | K | 37 | 5,756 | 4,429 | 4,601 |
| gas | semi | A | 38 | 14,311 | 15,103 | 16,254 |
| gas | semi | B | 32 | 13,489 | 15,987 | 17,106 |
| gas | semi | C | 35 | 12,903 | 12,524 | 13,442 |
| gas | semi | D | 41 | 12,082 | 11,285 | 12,080 |
| gas | semi | E | 36 | 11,495 | 9,710 | 10,279 |
| gas | semi | F | 20 | 11,144 | 9,907 | 10,460 |
| gas | semi | G | 26 | 10,557 | 8,926 | 9,478 |
| gas | semi | H | 23 | 10,205 | 9,295 | 9,929 |
| gas | semi | I | 19 | 9,853 | 7,920 | 8,436 |
| gas | semi | J | 20 | 9,384 | 6,354 | 6,679 |
| gas | semi | K | 34 | 9,032 | 6,466 | 6,806 |
| gas | terraced | A | 67 | 12,627 | 10,499 | 11,191 |
| gas | terraced | B | 47 | 11,902 | 10,450 | 11,264 |
| gas | terraced | C | 23 | 11,385 | 9,525 | 10,338 |
| gas | terraced | D | 22 | 10,660 | 9,245 | 9,840 |
| gas | terraced | E | 25 | 10,143 | 9,368 | 9,970 |
| gas | terraced | G | 19 | 9,315 | 8,418 | 8,930 |
| gas | terraced | K | 12 | 7,970 | 6,778 | 7,177 |

## EPC-style rating distribution vs EPC reference (placeholder shares)

| band | simulated | reference |
|---|---|---|
| A | 0.005 | 0.002 |
| B | 0.021 | 0.030 |
| C | 0.426 | 0.380 |
| D | 0.376 | 0.420 |
| E | 0.109 | 0.130 |
| F | 0.039 | 0.030 |
| G | 0.024 | 0.008 |

The monthly electricity shape uses the reference smart-meter shape directly (data/reference/parameters.json), so it matches by construction until a smart-meter subset is loaded.
