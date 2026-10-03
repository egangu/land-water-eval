# Data

Download the CSVs (`latitude,longitude,label`, coordinates in decimal degrees):

| Grid | Points | Download |
| --- | ---: | --- |
| 2° · 90 × 180 | 16,200 | [land_water_gt_2deg.csv](https://raw.githubusercontent.com/egangu/land-water-eval/main/data/land_water_gt_2deg.csv) |
| 1° · 180 × 360 | 64,800 | [land_water_gt_1deg.csv](https://raw.githubusercontent.com/egangu/land-water-eval/main/data/land_water_gt_1deg.csv) |

Both use cell centers and labels sampled directly from the unmodified
**GSHHG 2.3.7 intermediate (i)** shoreline polygons. Lakes are `Water`,
islands in lakes are `Land`, and the Antarctic ice front encloses `Land`.
Source: [official GSHHG data](https://www.soest.hawaii.edu/pwessel/gshhg/).

GSHHG is distributed under the GNU LGPL; its upstream license texts are included
as [COPYING.LESSER](COPYING.LESSER) and [COPYING](COPYING). The project's
[MIT License](../LICENSE) covers its original code and does not relicense these data.
