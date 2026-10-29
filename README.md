# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/geoff-davis/async-batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                         |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|------------------------------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| src/async\_batch\_llm/\_\_init\_\_.py                        |       50 |        2 |        4 |        0 |     96.30% |   356-358 |
| src/async\_batch\_llm/\_internal/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/admission.py                |      421 |       16 |      118 |       14 |     94.43% |106, 114, 131, 133, 135, 155, 171, 173, 267, 299-300, 389, 403, 408-409, 523-\>521, 605 |
| src/async\_batch\_llm/\_internal/artifact\_codec.py          |      193 |       14 |       58 |        7 |     91.63% |69, 73, 77, 124, 153, 189, 300-\>299, 376, 402, 410, 455, 471-474 |
| src/async\_batch\_llm/\_internal/backoff.py                  |       14 |        2 |        8 |        0 |     90.91% |     19-20 |
| src/async\_batch\_llm/\_internal/budget.py                   |       74 |        0 |       24 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/capacity.py                 |      131 |       11 |       32 |        2 |     90.80% |43-44, 84-93, 159-160, 187-193 |
| src/async\_batch\_llm/\_internal/classifier\_resolver.py     |       38 |        0 |        4 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/cleanup.py                  |      262 |        8 |       94 |        7 |     94.66% |144-\>exit, 150, 177, 297-\>299, 303-\>exit, 384, 417-\>430, 471-475 |
| src/async\_batch\_llm/\_internal/error\_logging.py           |      107 |       16 |       40 |        8 |     82.31% |42-\>45, 93, 107-\>117, 115-116, 118, 121-132, 149-\>170, 151, 153-156, 167-168 |
| src/async\_batch\_llm/\_internal/event\_dispatcher.py        |       90 |        4 |       30 |        1 |     95.83% |82-83, 122-\>124, 147, 163 |
| src/async\_batch\_llm/\_internal/execution\_state.py         |       59 |        1 |        6 |        1 |     96.92% |       125 |
| src/async\_batch\_llm/\_internal/executor\_host.py           |       64 |        0 |        4 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/guardrails.py               |      145 |        4 |       50 |        6 |     94.87% |79, 89, 121, 160-\>exit, 178, 192-\>194 |
| src/async\_batch\_llm/\_internal/input\_validation.py        |       43 |        1 |       12 |        1 |     96.36% |        34 |
| src/async\_batch\_llm/\_internal/item\_executor.py           |      768 |       23 |      234 |       33 |     94.41% |202, 244-\>exit, 246-\>exit, 250-\>exit, 259-\>exit, 267-\>exit, 301-\>exit, 318, 334, 410, 470, 530-\>exit, 539-\>541, 542-\>544, 558-561, 650, 676-\>684, 723-\>727, 741-\>exit, 785, 787, 941, 1016, 1091-1098, 1138, 1362-1363, 1366-\>1371, 1384, 1473-\>exit, 1535, 1551-\>1555, 1572-\>1576, 1626-\>1630, 1635-\>1639, 1674-\>1679, 1722-\>1733, 1819-\>1826, 1875, 1877-\>1879 |
| src/async\_batch\_llm/\_internal/logical\_item.py            |       17 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/rate\_limit\_coordinator.py |      168 |        5 |       42 |        1 |     97.14% |122, 293, 370-376 |
| src/async\_batch\_llm/\_internal/responses\_translation.py   |      163 |       13 |      102 |       15 |     89.43% |50, 69, 97, 106, 119, 121-\>123, 139, 144, 157, 168, 170-\>172, 189, 208, 212, 222 |
| src/async\_batch\_llm/\_internal/strategy\_lifecycle.py      |      172 |        5 |       42 |        5 |     95.33% |91, 143-\>141, 218, 245, 251-\>254, 257, 330 |
| src/async\_batch\_llm/artifacts.py                           |      539 |       70 |      176 |       33 |     84.76% |131, 143-\>145, 160, 162-164, 199-200, 261-262, 278-279, 286, 290-291, 307-308, 323, 328, 341, 345, 350, 364-365, 369-\>373, 388, 390-393, 395, 418-419, 428, 434-438, 443, 449, 459, 465, 482-483, 487, 543-544, 551, 588, 606-607, 654-655, 680, 760-761, 781, 814, 817, 855-857, 861-862, 879, 894, 911-\>923, 916-917, 945, 954-\>956, 968-969, 978-979, 1013-\>1015 |
| src/async\_batch\_llm/base.py                                |     1002 |       42 |      278 |       21 |     94.45% |122, 350, 359, 435-\>exit, 778-\>exit, 781-\>exit, 835, 872-\>874, 925-\>930, 1263, 1265, 1642, 1684-1685, 1851, 1947-1952, 1977, 1992-1995, 2016-\>exit, 2053-2054, 2084, 2088, 2131-\>exit, 2142, 2170-2171, 2193-2201, 2217-\>2224, 2297, 2346, 2365, 2369-\>2375, 2371-2374, 2405-2406, 2421-2423 |
| src/async\_batch\_llm/budget.py                              |       13 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/call\_pool.py                          |       80 |        0 |       14 |        0 |    100.00% |           |
| src/async\_batch\_llm/callable\_strategy.py                  |      144 |        4 |       50 |        6 |     94.85% |57-\>exit, 70-\>exit, 89, 132, 173, 247 |
| src/async\_batch\_llm/categories.py                          |       92 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/classifiers/\_\_init\_\_.py            |        5 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/classifiers/gemini.py                  |       92 |        7 |       50 |        5 |     91.55% |27-\>30, 41, 55-\>52, 60-61, 115-116, 223, 231 |
| src/async\_batch\_llm/classifiers/openai.py                  |       71 |        5 |       40 |        1 |     94.59% |114, 146-147, 181-182 |
| src/async\_batch\_llm/classifiers/openrouter.py              |       37 |        2 |       16 |        0 |     96.23% |     85-86 |
| src/async\_batch\_llm/classifiers/pydantic\_ai.py            |       16 |        2 |        6 |        0 |     90.91% |     16-17 |
| src/async\_batch\_llm/core/\_\_init\_\_.py                   |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/config.py                         |      263 |        4 |      140 |        4 |     98.01% |153, 155, 268, 565 |
| src/async\_batch\_llm/core/protocols.py                      |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/factory.py                             |       66 |        3 |       32 |        3 |     91.84% |108-\>exit, 119-\>exit, 221-226 |
| src/async\_batch\_llm/gateway.py                             |       11 |        0 |        2 |        0 |    100.00% |           |
| src/async\_batch\_llm/llm\_strategies.py                     |      204 |       11 |       54 |        9 |     92.25% |36-37, 71-\>69, 73, 88-90, 335-\>exit, 347-\>exit, 434, 452, 465-\>468, 636-\>638, 641, 782-784 |
| src/async\_batch\_llm/middleware/\_\_init\_\_.py             |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/middleware/base.py                     |       11 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/models.py                              |     1144 |       99 |      508 |       79 |     88.26% |55-57, 60-61, 111-127, 393-\>396, 465-\>exit, 473, 476, 514, 521, 544, 711-\>714, 716, 722-\>exit, 747, 828, 831, 834, 877-881, 905, 916, 942-943, 959-\>993, 970, 972, 1008, 1024-\>1042, 1063-\>1068, 1339-\>1388, 1342-\>1345, 1346-\>1382, 1362-1363, 1368, 1389-\>1393, 1391-1392, 1403, 1457-\>1467, 1460-\>1467, 1462-1463, 1483, 1486, 1489-\>1497, 1699, 1708-1713, 1746, 1774-1776, 1814, 1827, 1844-\>1846, 1847-\>1849, 1850-\>1852, 1864-1867, 1869-1873, 1881, 1883, 1946-1947, 2125, 2128-\>2130, 2131-\>2133, 2140, 2145-\>2150, 2157-\>2159, 2184-\>2182, 2188-\>2186, 2192-\>2179, 2202-\>2179, 2207-\>2203, 2209-\>2203, 2212-\>2210, 2229, 2280-2281, 2367, 2390, 2400-2401, 2404, 2416-\>2425, 2420-2424, 2431-2432, 2445, 2518, 2527, 2531-2532, 2566, 2569, 2572-2575, 2698, 2700, 2720-\>2725 |
| src/async\_batch\_llm/observers/\_\_init\_\_.py              |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/observers/base.py                      |       25 |        1 |        0 |        0 |     96.00% |        63 |
| src/async\_batch\_llm/observers/metrics.py                   |      143 |        5 |       72 |       10 |     92.09% |56, 73-\>80, 83-\>89, 90-\>exit, 115-\>109, 118-\>exit, 129, 135-136, 153, 176-\>exit |
| src/async\_batch\_llm/parallel.py                            |      448 |       22 |      120 |        7 |     94.89% |112-113, 116-117, 347, 358, 362, 370, 374, 378, 382, 386, 398, 431-\>433, 528, 542, 550, 583-584, 627-\>630, 693, 698, 793, 976-\>981, 1028 |
| src/async\_batch\_llm/parsing.py                             |       63 |        0 |       18 |        0 |    100.00% |           |
| src/async\_batch\_llm/provider\_output.py                    |      102 |        1 |       28 |        1 |     98.46% |       225 |
| src/async\_batch\_llm/serialization.py                       |      250 |       39 |      102 |       21 |     82.39% |117, 131, 140-141, 223-224, 228, 253, 259, 269, 271, 351-352, 367, 377-378, 387, 389, 403, 474, 481, 488, 493, 522, 539, 575, 606-609, 661-662, 673-674, 680, 683-684, 696, 707 |
| src/async\_batch\_llm/single.py                              |       35 |        3 |        4 |        1 |     89.74% | 54-55, 69 |
| src/async\_batch\_llm/sqlite\_artifacts.py                   |      721 |       89 |      208 |       39 |     84.93% |88-89, 126-\>exit, 134, 139, 218, 226-227, 276-277, 284-286, 291-296, 319, 361-362, 421-422, 427, 431, 461, 528-531, 551, 583, 602-605, 608-609, 611-\>618, 614-615, 626, 655, 663, 677, 683, 689-690, 712, 714-\>exit, 728-\>730, 731, 737-\>739, 748, 756-762, 799-800, 811-\>810, 826-\>825, 840-841, 853, 863-865, 881-\>896, 910, 930, 947-950, 954-956, 1049-1051, 1067, 1134, 1141, 1144, 1148, 1159, 1190, 1286, 1352-1353, 1399-1400, 1408-1409, 1412-1413, 1417 |
| src/async\_batch\_llm/strategies/\_\_init\_\_.py             |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/strategies/errors.py                   |      234 |       18 |       84 |        3 |     93.40% |90-\>94, 131-132, 137-138, 144-155, 306, 551-552, 589 |
| src/async\_batch\_llm/strategies/rate\_limit.py              |       32 |        0 |        2 |        0 |    100.00% |           |
| src/async\_batch\_llm/streaming.py                           |      217 |        2 |       80 |        5 |     97.64% |81-\>exit, 109, 218-\>220, 274-\>277, 370, 376-\>exit |
| src/async\_batch\_llm/token\_estimation.py                   |       35 |        0 |        4 |        0 |    100.00% |           |
| src/async\_batch\_llm/token\_extractor.py                    |      136 |        3 |       56 |        5 |     95.83% |88-\>98, 101-\>107, 103-\>107, 146-147, 154, 206-\>208 |
| **TOTAL**                                                    | **9223** |  **557** | **3048** |  **354** | **92.12%** |           |


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