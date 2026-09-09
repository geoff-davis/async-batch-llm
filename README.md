# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/geoff-davis/async-batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                         |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|------------------------------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| src/async\_batch\_llm/\_\_init\_\_.py                        |       36 |        2 |        0 |        0 |     94.44% |   353-355 |
| src/async\_batch\_llm/\_internal/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/admission.py                |      409 |       17 |      116 |       14 |     94.10% |102, 106, 114, 131, 133, 135, 155, 171, 173, 261, 293-294, 378, 392, 397-398, 509-\>507, 575 |
| src/async\_batch\_llm/\_internal/artifact\_codec.py          |      150 |       13 |       36 |        6 |     89.78% |56, 60, 64, 93, 127, 293, 319, 327, 372, 388-391 |
| src/async\_batch\_llm/\_internal/capacity.py                 |      131 |       11 |       32 |        2 |     90.80% |43-44, 84-93, 159-160, 187-193 |
| src/async\_batch\_llm/\_internal/classifier\_resolver.py     |       38 |        0 |        4 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/cleanup.py                  |      261 |        8 |       94 |        7 |     94.65% |142-\>exit, 148, 175, 295-\>297, 301-\>exit, 382, 415-\>428, 469-473 |
| src/async\_batch\_llm/\_internal/error\_logging.py           |       76 |       16 |       28 |        8 |     75.00% |36-\>39, 87, 101-\>111, 109-110, 112, 115-126, 143-\>164, 145, 147-150, 161-162 |
| src/async\_batch\_llm/\_internal/event\_dispatcher.py        |       81 |        5 |       22 |        2 |     93.20% |78-79, 118-\>120, 133, 147, 149 |
| src/async\_batch\_llm/\_internal/execution\_state.py         |       38 |        1 |        6 |        1 |     95.45% |        75 |
| src/async\_batch\_llm/\_internal/executor\_host.py           |       60 |        0 |        2 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/guardrails.py               |      136 |        9 |       44 |        5 |     91.11% |59, 69, 98, 137-\>exit, 155, 167-171 |
| src/async\_batch\_llm/\_internal/item\_executor.py           |      647 |       32 |      212 |       43 |     91.27% |172-\>174, 206-\>exit, 208-\>exit, 212-\>exit, 221-\>exit, 229-\>exit, 257, 273, 342, 402, 437-440, 447-448, 460, 517, 543-\>551, 588-\>592, 606-\>exit, 650, 652, 782, 801, 808-\>818, 831, 887-892, 913-\>916, 931-\>exit, 941, 1072, 1095-\>exit, 1151, 1179, 1195-\>1199, 1239-1240, 1243-\>1248, 1280-\>1288, 1328-\>1332, 1337-\>1341, 1376-\>1405, 1395-\>1405, 1416-\>1427, 1427-\>1432, 1434-\>1445, 1539, 1546-\>1548, 1560, 1562, 1564-\>1566, 1571, 1582-\>1584, 1677, 1684-\>1687 |
| src/async\_batch\_llm/\_internal/logical\_item.py            |       16 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/rate\_limit\_coordinator.py |      157 |        5 |       36 |        1 |     96.89% |117, 277, 349-355 |
| src/async\_batch\_llm/\_internal/strategy\_lifecycle.py      |       69 |        4 |       14 |        2 |     92.77% |66, 92, 100, 167 |
| src/async\_batch\_llm/artifacts.py                           |      462 |       72 |      150 |       32 |     82.35% |121, 132-\>134, 142, 144-146, 180-181, 242-243, 251-252, 259, 263-264, 283, 290, 296, 300, 305, 316, 318-321, 323, 346, 351, 366, 371-376, 382, 388, 405-406, 410, 466-467, 470-471, 473, 509, 527-528, 575-576, 601, 654-659, 668, 679, 711, 714, 750-751, 760, 774, 785, 791-\>803, 796-797, 825, 832-835, 846-847, 856-857, 863 |
| src/async\_batch\_llm/base.py                                |      906 |       47 |      244 |       23 |     93.22% |119, 333, 342, 419-\>exit, 708-\>exit, 711-\>exit, 762, 799-\>801, 854-\>859, 1136, 1138, 1471, 1513-1514, 1670, 1708, 1760-1765, 1790, 1805-1808, 1829-\>exit, 1866-1867, 1897, 1901, 1940-\>exit, 1951, 1974, 1976-1980, 2002-2010, 2104, 2153, 2172, 2176-\>2182, 2178-2181, 2212-2213, 2228-2230 |
| src/async\_batch\_llm/callable\_strategy.py                  |      131 |        4 |       46 |        6 |     94.35% |50-\>exit, 63-\>exit, 82, 122, 163, 222 |
| src/async\_batch\_llm/classifiers/\_\_init\_\_.py            |        4 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/classifiers/gemini.py                  |       69 |        9 |       38 |        5 |     86.92% |51, 53, 55, 84-85, 119-120, 222, 230 |
| src/async\_batch\_llm/classifiers/openai.py                  |       76 |       10 |       46 |        6 |     86.89% |70, 72, 74, 116-117, 125, 135, 201-202, 290 |
| src/async\_batch\_llm/classifiers/openrouter.py              |       19 |        0 |        8 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/\_\_init\_\_.py                   |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/config.py                         |      184 |        3 |       92 |        3 |     97.83% |205, 440, 448 |
| src/async\_batch\_llm/core/protocols.py                      |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/factory.py                             |       45 |        0 |       22 |        2 |     97.01% |63-\>exit, 74-\>exit |
| src/async\_batch\_llm/gateway.py                             |       79 |        0 |       14 |        0 |    100.00% |           |
| src/async\_batch\_llm/llm\_strategies.py                     |      179 |       13 |       40 |        8 |     90.41% |33-34, 68-\>66, 70, 85-87, 279-280, 330-\>exit, 342-\>exit, 425, 443, 606-\>608, 611, 739-741 |
| src/async\_batch\_llm/middleware/\_\_init\_\_.py             |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/middleware/base.py                     |       11 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/models.py                              |      784 |      104 |      346 |       59 |     82.74% |48-50, 53-54, 283-\>286, 363-\>366, 371, 374, 394, 539-\>542, 544, 550-\>exit, 575, 654-\>657, 658, 661, 674-678, 714-715, 731-\>761, 742, 773, 807, 1079-\>1128, 1082-\>1085, 1086-\>1122, 1102-1103, 1108, 1129-\>1133, 1131-1132, 1176-\>1186, 1179-\>1186, 1181-1182, 1200, 1203-\>exit, 1402-1403, 1489, 1512, 1522-1523, 1526, 1538-\>1547, 1542-1546, 1552-1560, 1567, 1641, 1650, 1685, 1688, 1690-1694, 1771-\>1773, 1779-1780, 1817, 1819, 1851-\>1853, 1869, 1871, 1881, 1887-1900, 1902-1904, 1911-1919, 1921, 1936-1937, 1947, 1958-\>1960, 1961-\>1963, 1964-\>1966, 1966-\>1973, 1976-1987, 1995, 1997, 2010-\>2015 |
| src/async\_batch\_llm/observers/\_\_init\_\_.py              |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/observers/base.py                      |       25 |        1 |        0 |        0 |     96.00% |        63 |
| src/async\_batch\_llm/observers/metrics.py                   |      139 |       12 |       70 |       14 |     85.65% |55, 72-\>79, 82-\>88, 89-\>exit, 114-\>108, 117-\>exit, 128, 134-135, 141, 142-\>147, 152, 155-160, 163, 169-\>exit |
| src/async\_batch\_llm/parallel.py                            |      413 |       27 |      110 |       10 |     92.93% |102-103, 106-107, 174, 214, 322, 333, 337, 345, 349, 353, 357, 361, 373, 480, 504-507, 545-546, 588-\>591, 654, 659, 754, 819, 876-\>881, 905-\>910, 957 |
| src/async\_batch\_llm/parsing.py                             |       63 |        0 |       18 |        0 |    100.00% |           |
| src/async\_batch\_llm/provider\_output.py                    |      102 |        1 |       28 |        1 |     98.46% |       225 |
| src/async\_batch\_llm/serialization.py                       |      249 |       41 |      102 |       23 |     81.20% |113, 116, 130, 139-140, 170, 222-223, 227, 252, 258, 268, 270, 350-351, 366, 376-377, 386, 388, 402, 473, 480, 487, 492, 510, 527, 563, 594-597, 649-650, 661-662, 668, 671-672, 684, 695 |
| src/async\_batch\_llm/single.py                              |       32 |        3 |        4 |        1 |     88.89% | 52-53, 67 |
| src/async\_batch\_llm/sqlite\_artifacts.py                   |      692 |       87 |      206 |       40 |     84.52% |86-87, 123-\>exit, 131, 136, 211-212, 214, 222-223, 264-265, 290, 300, 332-333, 365-370, 375, 379, 409, 418-419, 476-479, 499, 531, 550-553, 556-557, 559-\>566, 562-563, 574, 603, 611, 625, 631, 637-638, 660, 662-\>exit, 676-\>678, 679, 685-\>687, 696, 704-710, 747-748, 759-\>758, 774-\>773, 788-789, 801, 811-813, 829-\>844, 858, 878, 895-898, 902-904, 997-999, 1015, 1082, 1089, 1092, 1096, 1107, 1138, 1234, 1300-1301, 1347-1348, 1356-1357, 1360-1361, 1365 |
| src/async\_batch\_llm/strategies/\_\_init\_\_.py             |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/strategies/errors.py                   |      135 |       17 |       36 |        6 |     86.55% |62-73, 154-\>exit, 187, 206, 368, 397, 405, 459-460 |
| src/async\_batch\_llm/strategies/rate\_limit.py              |       31 |        0 |        2 |        0 |    100.00% |           |
| src/async\_batch\_llm/streaming.py                           |      199 |        1 |       70 |        4 |     98.14% |65-\>exit, 197-\>199, 253-\>256, 330, 336-\>exit |
| src/async\_batch\_llm/token\_estimation.py                   |       35 |        0 |        4 |        0 |    100.00% |           |
| src/async\_batch\_llm/token\_extractor.py                    |      111 |       10 |       48 |        9 |     86.79% |73-\>81, 88-\>92, 96-\>109, 119, 149, 162, 173-\>176, 185-188, 214, 235, 246-247 |
| **TOTAL**                                                    | **7493** |  **585** | **2390** |  **343** | **89.86%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/geoff-davis/async-batch-llm/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/geoff-davis/async-batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/geoff-davis/async-batch-llm/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/geoff-davis/async-batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fgeoff-davis%2Fasync-batch-llm%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/geoff-davis/async-batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.