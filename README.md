# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/geoff-davis/async-batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                         |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|------------------------------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| src/async\_batch\_llm/\_\_init\_\_.py                        |       49 |        2 |        4 |        0 |     96.23% |   353-355 |
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
| src/async\_batch\_llm/\_internal/input\_validation.py        |       32 |        1 |        8 |        1 |     95.00% |        33 |
| src/async\_batch\_llm/\_internal/item\_executor.py           |      768 |       23 |      234 |       33 |     94.41% |202, 244-\>exit, 246-\>exit, 250-\>exit, 259-\>exit, 267-\>exit, 301-\>exit, 318, 334, 410, 470, 530-\>exit, 539-\>541, 542-\>544, 558-561, 650, 676-\>684, 723-\>727, 741-\>exit, 785, 787, 941, 1016, 1091-1098, 1138, 1356-1357, 1360-\>1365, 1378, 1467-\>exit, 1529, 1545-\>1549, 1566-\>1570, 1620-\>1624, 1629-\>1633, 1668-\>1673, 1716-\>1727, 1813-\>1820, 1869, 1871-\>1873 |
| src/async\_batch\_llm/\_internal/logical\_item.py            |       17 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/rate\_limit\_coordinator.py |      168 |        5 |       42 |        1 |     97.14% |122, 293, 370-376 |
| src/async\_batch\_llm/\_internal/responses\_translation.py   |      163 |       13 |      102 |       15 |     89.43% |50, 69, 97, 106, 119, 121-\>123, 139, 144, 157, 168, 170-\>172, 189, 208, 212, 222 |
| src/async\_batch\_llm/\_internal/strategy\_lifecycle.py      |      172 |        5 |       42 |        5 |     95.33% |91, 143-\>141, 218, 245, 251-\>254, 257, 330 |
| src/async\_batch\_llm/artifacts.py                           |      539 |       70 |      176 |       34 |     84.62% |131, 143-\>145, 160, 162-164, 199-200, 261-262, 278-279, 286, 290-291, 307-308, 323, 328, 341, 345, 350, 364-365, 369-\>373, 388, 390-393, 395, 418-419, 428, 434-438, 443, 449, 459, 465, 482-483, 487, 543-544, 551, 588, 606-607, 654-655, 680, 760-761, 781, 787-\>786, 814, 817, 855-857, 861-862, 879, 894, 911-\>923, 916-917, 945, 954-\>956, 968-969, 978-979, 1013-\>1015 |
| src/async\_batch\_llm/base.py                                |      982 |       42 |      272 |       21 |     94.34% |122, 338, 347, 423-\>exit, 720-\>exit, 723-\>exit, 774, 811-\>813, 866-\>871, 1194, 1196, 1573, 1615-1616, 1782, 1878-1883, 1908, 1923-1926, 1947-\>exit, 1984-1985, 2015, 2019, 2062-\>exit, 2073, 2101-2102, 2124-2132, 2148-\>2155, 2228, 2277, 2296, 2300-\>2306, 2302-2305, 2336-2337, 2352-2354 |
| src/async\_batch\_llm/budget.py                              |       13 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/callable\_strategy.py                  |      144 |        4 |       50 |        6 |     94.85% |51-\>exit, 64-\>exit, 83, 126, 167, 241 |
| src/async\_batch\_llm/categories.py                          |       92 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/classifiers/\_\_init\_\_.py            |        5 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/classifiers/gemini.py                  |       92 |        7 |       50 |        5 |     91.55% |27-\>30, 41, 55-\>52, 60-61, 115-116, 223, 231 |
| src/async\_batch\_llm/classifiers/openai.py                  |       71 |        5 |       40 |        1 |     94.59% |114, 146-147, 181-182 |
| src/async\_batch\_llm/classifiers/openrouter.py              |       37 |        2 |       16 |        0 |     96.23% |     85-86 |
| src/async\_batch\_llm/classifiers/pydantic\_ai.py            |       16 |        2 |        6 |        0 |     90.91% |     16-17 |
| src/async\_batch\_llm/core/\_\_init\_\_.py                   |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/config.py                         |      246 |        4 |      138 |        4 |     97.92% |151, 153, 266, 553 |
| src/async\_batch\_llm/core/protocols.py                      |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/factory.py                             |       66 |        3 |       32 |        3 |     91.84% |108-\>exit, 119-\>exit, 221-226 |
| src/async\_batch\_llm/gateway.py                             |       87 |        0 |       16 |        0 |    100.00% |           |
| src/async\_batch\_llm/llm\_strategies.py                     |      200 |       11 |       50 |        9 |     92.00% |36-37, 71-\>69, 73, 88-90, 335-\>exit, 347-\>exit, 434, 452, 465-\>468, 636-\>638, 641, 774-776 |
| src/async\_batch\_llm/middleware/\_\_init\_\_.py             |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/middleware/base.py                     |       11 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/models.py                              |     1125 |      100 |      500 |       80 |     87.94% |55-57, 60-61, 111-127, 364-\>367, 436-\>exit, 444, 447, 485, 492, 515, 682-\>685, 687, 693-\>exit, 718, 799, 802, 805, 848-852, 876, 887, 913-914, 930-\>964, 941, 943, 979, 995-\>1013, 1034-\>1039, 1295-\>1344, 1298-\>1301, 1302-\>1338, 1318-1319, 1324, 1345-\>1349, 1347-1348, 1359, 1413-\>1423, 1416-\>1423, 1418-1419, 1439, 1442, 1445-\>1453, 1655, 1664-1669, 1702, 1729-1731, 1738, 1765, 1782, 1799-\>1801, 1802-\>1804, 1805-\>1807, 1819-1822, 1824-1828, 1836, 1838, 1901-1902, 2074, 2077-\>2079, 2080-\>2082, 2089, 2094-\>2099, 2106-\>2108, 2133-\>2131, 2137-\>2135, 2141-\>2128, 2151-\>2128, 2156-\>2152, 2158-\>2152, 2161-\>2159, 2178, 2229-2230, 2316, 2339, 2349-2350, 2353, 2365-\>2374, 2369-2373, 2380-2381, 2394, 2467, 2476, 2480-2481, 2515, 2518, 2521-2524, 2647, 2649, 2669-\>2674 |
| src/async\_batch\_llm/observers/\_\_init\_\_.py              |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/observers/base.py                      |       25 |        1 |        0 |        0 |     96.00% |        63 |
| src/async\_batch\_llm/observers/metrics.py                   |      143 |        5 |       72 |       10 |     92.09% |56, 73-\>80, 83-\>89, 90-\>exit, 115-\>109, 118-\>exit, 129, 135-136, 153, 176-\>exit |
| src/async\_batch\_llm/parallel.py                            |      446 |       24 |      120 |        9 |     94.17% |111-112, 115-116, 183, 223, 346, 357, 361, 369, 373, 377, 381, 385, 397, 430-\>432, 527, 541, 549, 582-583, 626-\>629, 692, 697, 792, 972-\>977, 1024 |
| src/async\_batch\_llm/parsing.py                             |       63 |        0 |       18 |        0 |    100.00% |           |
| src/async\_batch\_llm/provider\_output.py                    |      102 |        1 |       28 |        1 |     98.46% |       225 |
| src/async\_batch\_llm/serialization.py                       |      250 |       39 |      102 |       21 |     82.39% |117, 131, 140-141, 223-224, 228, 253, 259, 269, 271, 351-352, 367, 377-378, 387, 389, 403, 474, 481, 488, 493, 522, 539, 575, 606-609, 661-662, 673-674, 680, 683-684, 696, 707 |
| src/async\_batch\_llm/single.py                              |       35 |        3 |        4 |        1 |     89.74% | 54-55, 69 |
| src/async\_batch\_llm/sqlite\_artifacts.py                   |      721 |       88 |      208 |       37 |     85.25% |88-89, 134, 218, 226-227, 276-277, 284-286, 291-296, 319, 361-362, 421-422, 427, 431, 461, 528-531, 551, 583, 602-605, 608-609, 611-\>618, 614-615, 626, 655, 663, 677, 683, 689-690, 712, 714-\>exit, 728-\>730, 731, 737-\>739, 748, 756-762, 799-800, 811-\>810, 826-\>825, 840-841, 853, 863-865, 881-\>896, 910, 930, 947-950, 954-956, 1049-1051, 1067, 1134, 1141, 1144, 1148, 1159, 1190, 1286, 1352-1353, 1399-1400, 1408-1409, 1412-1413, 1417 |
| src/async\_batch\_llm/strategies/\_\_init\_\_.py             |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/strategies/errors.py                   |      234 |       18 |       84 |        3 |     93.40% |90-\>94, 131-132, 137-138, 144-155, 306, 551-552, 589 |
| src/async\_batch\_llm/strategies/rate\_limit.py              |       32 |        0 |        2 |        0 |    100.00% |           |
| src/async\_batch\_llm/streaming.py                           |      217 |        2 |       80 |        5 |     97.64% |81-\>exit, 109, 218-\>220, 274-\>277, 370, 376-\>exit |
| src/async\_batch\_llm/token\_estimation.py                   |       35 |        0 |        4 |        0 |    100.00% |           |
| src/async\_batch\_llm/token\_extractor.py                    |      128 |        4 |       50 |        5 |     94.94% |85-\>95, 98-\>104, 100-\>104, 143-144, 151, 198-\>200, 214 |
| **TOTAL**                                                    | **9137** |  **560** | **3018** |  **356** | **92.00%** |           |


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