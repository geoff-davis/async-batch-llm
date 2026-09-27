# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/geoff-davis/async-batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                         |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|------------------------------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| src/async\_batch\_llm/\_\_init\_\_.py                        |       36 |        2 |        0 |        0 |     94.44% |   359-361 |
| src/async\_batch\_llm/\_internal/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/admission.py                |      421 |       16 |      118 |       14 |     94.43% |106, 114, 131, 133, 135, 155, 171, 173, 267, 299-300, 389, 403, 408-409, 523-\>521, 605 |
| src/async\_batch\_llm/\_internal/artifact\_codec.py          |      191 |       14 |       58 |        7 |     91.57% |60, 64, 68, 115, 144, 180, 291-\>290, 367, 393, 401, 446, 462-465 |
| src/async\_batch\_llm/\_internal/backoff.py                  |       14 |        2 |        8 |        0 |     90.91% |     19-20 |
| src/async\_batch\_llm/\_internal/capacity.py                 |      131 |       11 |       32 |        2 |     90.80% |43-44, 84-93, 159-160, 187-193 |
| src/async\_batch\_llm/\_internal/classifier\_resolver.py     |       38 |        0 |        4 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/cleanup.py                  |      261 |        8 |       94 |        7 |     94.65% |142-\>exit, 148, 175, 295-\>297, 301-\>exit, 382, 415-\>428, 469-473 |
| src/async\_batch\_llm/\_internal/error\_logging.py           |       76 |       16 |       28 |        8 |     75.00% |36-\>39, 87, 101-\>111, 109-110, 112, 115-126, 143-\>164, 145, 147-150, 161-162 |
| src/async\_batch\_llm/\_internal/event\_dispatcher.py        |       90 |        4 |       30 |        1 |     95.83% |82-83, 122-\>124, 147, 163 |
| src/async\_batch\_llm/\_internal/execution\_state.py         |       59 |        1 |        6 |        1 |     96.92% |       125 |
| src/async\_batch\_llm/\_internal/executor\_host.py           |       60 |        0 |        2 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/guardrails.py               |      137 |        4 |       44 |        6 |     94.48% |68, 78, 107, 146-\>exit, 164, 178-\>180 |
| src/async\_batch\_llm/\_internal/item\_executor.py           |      709 |       21 |      212 |       31 |     94.35% |228-\>exit, 230-\>exit, 234-\>exit, 243-\>exit, 251-\>exit, 279, 295, 367, 427, 473-\>exit, 482-\>484, 485-\>487, 501-504, 561, 587-\>595, 634-\>638, 652-\>exit, 696, 698, 851, 926, 1001-1008, 1048, 1266-1267, 1270-\>1275, 1335-\>1341, 1358-\>exit, 1420, 1436-\>1440, 1457-\>1461, 1507-\>1511, 1516-\>1520, 1554-\>1559, 1581-\>1592, 1668-\>1675, 1724, 1726-\>1728 |
| src/async\_batch\_llm/\_internal/logical\_item.py            |       17 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/rate\_limit\_coordinator.py |      168 |        5 |       42 |        1 |     97.14% |122, 293, 370-376 |
| src/async\_batch\_llm/\_internal/strategy\_lifecycle.py      |      172 |        5 |       42 |        5 |     95.33% |91, 143-\>141, 218, 245, 251-\>254, 257, 330 |
| src/async\_batch\_llm/artifacts.py                           |      534 |       70 |      174 |       33 |     84.60% |130, 141-\>143, 151, 153-155, 189-190, 251-252, 268-269, 276, 280-281, 297-298, 313, 318, 331, 335, 340, 354-355, 359-\>363, 378, 380-383, 385, 408-409, 418, 424-428, 433, 439, 449, 455, 472-473, 477, 533-534, 541, 578, 596-597, 644-645, 670, 750-751, 771, 804, 807, 845-847, 851-852, 869, 884, 901-\>913, 906-907, 935, 944-\>946, 958-959, 968-969, 1003-\>1005 |
| src/async\_batch\_llm/base.py                                |      954 |       45 |      260 |       22 |     93.82% |121, 336, 345, 422-\>exit, 711-\>exit, 714-\>exit, 765, 802-\>804, 857-\>862, 1139, 1141, 1518, 1560-1561, 1727, 1823-1828, 1853, 1868-1871, 1892-\>exit, 1929-1930, 1960, 1964, 2007-\>exit, 2018, 2043-2047, 2069-2077, 2093-\>2100, 2173, 2222, 2241, 2245-\>2251, 2247-2250, 2281-2282, 2297-2299 |
| src/async\_batch\_llm/callable\_strategy.py                  |      130 |        4 |       44 |        6 |     94.25% |51-\>exit, 64-\>exit, 83, 119, 160, 219 |
| src/async\_batch\_llm/classifiers/\_\_init\_\_.py            |        5 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/classifiers/gemini.py                  |       92 |        9 |       48 |        5 |     90.00% |23-\>26, 37, 51-\>48, 56-57, 111-112, 148-149, 227, 235 |
| src/async\_batch\_llm/classifiers/openai.py                  |       76 |        7 |       40 |        1 |     93.10% |85-86, 104, 145-146, 180-181 |
| src/async\_batch\_llm/classifiers/openrouter.py              |       32 |        3 |       12 |        1 |     90.91% | 71-72, 76 |
| src/async\_batch\_llm/classifiers/pydantic\_ai.py            |       16 |        2 |        6 |        0 |     90.91% |     16-17 |
| src/async\_batch\_llm/core/\_\_init\_\_.py                   |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/config.py                         |      224 |        4 |      126 |        4 |     97.71% |147, 149, 241, 501 |
| src/async\_batch\_llm/core/protocols.py                      |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/factory.py                             |       62 |        0 |       28 |        2 |     97.78% |100-\>exit, 111-\>exit |
| src/async\_batch\_llm/gateway.py                             |       79 |        0 |       14 |        0 |    100.00% |           |
| src/async\_batch\_llm/llm\_strategies.py                     |      200 |       13 |       50 |        9 |     91.20% |36-37, 71-\>69, 73, 88-90, 282-283, 335-\>exit, 347-\>exit, 432, 450, 463-\>466, 634-\>636, 639, 772-774 |
| src/async\_batch\_llm/middleware/\_\_init\_\_.py             |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/middleware/base.py                     |       11 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/models.py                              |      957 |      116 |      410 |       67 |     84.56% |52-54, 57-58, 108-124, 350-\>353, 422-\>exit, 430, 433, 470-\>473, 478, 481, 501, 668-\>671, 673, 679-\>exit, 704, 784-\>787, 788, 791, 834-838, 862, 873, 899-900, 916-\>950, 927, 929, 965, 981-\>999, 1020-\>1025, 1282-\>1331, 1285-\>1288, 1289-\>1325, 1305-1306, 1311, 1332-\>1336, 1334-1335, 1346, 1399-\>1409, 1402-\>1409, 1404-1405, 1425, 1428, 1431-\>1439, 1657-1658, 1744, 1767, 1777-1778, 1781, 1793-\>1802, 1797-1801, 1807-1815, 1822, 1895, 1904, 1908-1909, 1943, 1946, 1949-1952, 2029-\>2031, 2037-2038, 2075, 2077, 2111-\>2115, 2141, 2147-2160, 2162-2164, 2171-2179, 2181, 2196-2197, 2207, 2224-\>2226, 2227-\>2229, 2230-\>2232, 2244-2247, 2249-2253, 2261, 2263, 2276-\>2281 |
| src/async\_batch\_llm/observers/\_\_init\_\_.py              |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/observers/base.py                      |       25 |        1 |        0 |        0 |     96.00% |        63 |
| src/async\_batch\_llm/observers/metrics.py                   |      142 |        5 |       72 |       10 |     92.06% |55, 72-\>79, 82-\>88, 89-\>exit, 114-\>108, 117-\>exit, 128, 134-135, 152, 175-\>exit |
| src/async\_batch\_llm/parallel.py                            |      425 |       24 |      116 |        8 |     94.09% |109-110, 113-114, 181, 221, 334, 345, 349, 357, 361, 365, 369, 373, 385, 494, 508, 516, 549-550, 592-\>595, 658, 663, 758, 938-\>943, 990 |
| src/async\_batch\_llm/parsing.py                             |       63 |        0 |       18 |        0 |    100.00% |           |
| src/async\_batch\_llm/provider\_output.py                    |      102 |        1 |       28 |        1 |     98.46% |       225 |
| src/async\_batch\_llm/serialization.py                       |      249 |       39 |      102 |       21 |     82.34% |116, 130, 139-140, 222-223, 227, 252, 258, 268, 270, 350-351, 366, 376-377, 386, 388, 402, 473, 480, 487, 492, 521, 538, 574, 605-608, 660-661, 672-673, 679, 682-683, 695, 706 |
| src/async\_batch\_llm/single.py                              |       32 |        3 |        4 |        1 |     88.89% | 52-53, 67 |
| src/async\_batch\_llm/sqlite\_artifacts.py                   |      721 |       89 |      208 |       39 |     84.93% |88-89, 126-\>exit, 134, 139, 218, 226-227, 276-277, 284-286, 291-296, 319, 361-362, 421-422, 427, 431, 461, 528-531, 551, 583, 602-605, 608-609, 611-\>618, 614-615, 626, 655, 663, 677, 683, 689-690, 712, 714-\>exit, 728-\>730, 731, 737-\>739, 748, 756-762, 799-800, 811-\>810, 826-\>825, 840-841, 853, 863-865, 881-\>896, 910, 930, 947-950, 954-956, 1049-1051, 1067, 1134, 1141, 1144, 1148, 1159, 1190, 1286, 1352-1353, 1399-1400, 1408-1409, 1412-1413, 1417 |
| src/async\_batch\_llm/strategies/\_\_init\_\_.py             |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/strategies/errors.py                   |      169 |       16 |       50 |        2 |     91.78% |63-64, 71-82, 202, 221, 431-432 |
| src/async\_batch\_llm/strategies/rate\_limit.py              |       32 |        0 |        2 |        0 |    100.00% |           |
| src/async\_batch\_llm/streaming.py                           |      198 |        1 |       70 |        4 |     98.13% |71-\>exit, 203-\>205, 259-\>262, 336, 342-\>exit |
| src/async\_batch\_llm/token\_estimation.py                   |       35 |        0 |        4 |        0 |    100.00% |           |
| src/async\_batch\_llm/token\_extractor.py                    |      128 |        4 |       50 |        5 |     94.94% |85-\>95, 98-\>104, 100-\>104, 143-144, 151, 198-\>200, 214 |
| **TOTAL**                                                    | **8286** |  **565** | **2656** |  **324** | **91.27%** |           |


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