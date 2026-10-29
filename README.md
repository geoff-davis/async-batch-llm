# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/geoff-davis/async-batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                         |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|------------------------------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| src/async\_batch\_llm/\_\_init\_\_.py                        |       51 |        2 |        4 |        0 |     96.36% |   359-361 |
| src/async\_batch\_llm/\_internal/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/admission.py                |      421 |       16 |      118 |       14 |     94.43% |106, 114, 131, 133, 135, 155, 171, 173, 267, 299-300, 389, 403, 408-409, 523-\>521, 605 |
| src/async\_batch\_llm/\_internal/artifact\_codec.py          |      193 |       14 |       58 |        7 |     91.63% |69, 73, 77, 124, 153, 189, 300-\>299, 376, 402, 410, 455, 471-474 |
| src/async\_batch\_llm/\_internal/backoff.py                  |       14 |        2 |        8 |        0 |     90.91% |     19-20 |
| src/async\_batch\_llm/\_internal/budget.py                   |       74 |        0 |       24 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/cache\_groups.py            |       40 |        0 |        6 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/capacity.py                 |      131 |       11 |       32 |        2 |     90.80% |43-44, 84-93, 159-160, 187-193 |
| src/async\_batch\_llm/\_internal/classifier\_resolver.py     |       38 |        0 |        4 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/cleanup.py                  |      262 |        8 |       94 |        7 |     94.66% |144-\>exit, 150, 177, 297-\>299, 303-\>exit, 384, 417-\>430, 471-475 |
| src/async\_batch\_llm/\_internal/error\_logging.py           |      107 |       16 |       40 |        8 |     82.31% |42-\>45, 93, 107-\>117, 115-116, 118, 121-132, 149-\>170, 151, 153-156, 167-168 |
| src/async\_batch\_llm/\_internal/event\_dispatcher.py        |       90 |        4 |       30 |        1 |     95.83% |82-83, 122-\>124, 147, 163 |
| src/async\_batch\_llm/\_internal/execution\_state.py         |       59 |        1 |        6 |        1 |     96.92% |       125 |
| src/async\_batch\_llm/\_internal/executor\_host.py           |       69 |        0 |        6 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/guardrails.py               |      145 |        4 |       50 |        6 |     94.87% |79, 89, 121, 160-\>exit, 178, 192-\>194 |
| src/async\_batch\_llm/\_internal/input\_validation.py        |       43 |        1 |       12 |        1 |     96.36% |        34 |
| src/async\_batch\_llm/\_internal/item\_executor.py           |      790 |       23 |      242 |       33 |     94.57% |203, 246-\>exit, 248-\>exit, 252-\>exit, 261-\>exit, 269-\>exit, 303-\>exit, 320, 336, 416, 476, 536-\>exit, 545-\>547, 548-\>550, 564-567, 681, 707-\>715, 754-\>758, 772-\>exit, 816, 818, 972, 1047, 1122-1129, 1169, 1393-1394, 1397-\>1402, 1415, 1504-\>exit, 1566, 1582-\>1586, 1603-\>1607, 1658-\>1662, 1667-\>1671, 1707-\>1712, 1766-\>1777, 1866-\>1873, 1922, 1924-\>1926 |
| src/async\_batch\_llm/\_internal/logical\_item.py            |       17 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/rate\_limit\_coordinator.py |      168 |        5 |       42 |        1 |     97.14% |122, 293, 370-376 |
| src/async\_batch\_llm/\_internal/responses\_translation.py   |      163 |       13 |      102 |       15 |     89.43% |50, 69, 97, 106, 119, 121-\>123, 139, 144, 157, 168, 170-\>172, 189, 208, 212, 222 |
| src/async\_batch\_llm/\_internal/strategy\_lifecycle.py      |      172 |        5 |       42 |        5 |     95.33% |91, 143-\>141, 218, 245, 251-\>254, 257, 330 |
| src/async\_batch\_llm/artifacts.py                           |      539 |       70 |      176 |       34 |     84.62% |131, 143-\>145, 160, 162-164, 199-200, 261-262, 278-279, 286, 290-291, 307-308, 323, 328, 341, 345, 350, 364-365, 369-\>373, 388, 390-393, 395, 418-419, 428, 434-438, 443, 449, 459, 465, 482-483, 487, 543-544, 551, 588, 606-607, 654-655, 680, 760-761, 781, 787-\>786, 814, 817, 855-857, 861-862, 879, 894, 911-\>923, 916-917, 945, 954-\>956, 968-969, 978-979, 1013-\>1015 |
| src/async\_batch\_llm/base.py                                |     1002 |       42 |      278 |       21 |     94.45% |122, 350, 359, 435-\>exit, 778-\>exit, 781-\>exit, 835, 872-\>874, 925-\>930, 1263, 1265, 1642, 1684-1685, 1851, 1947-1952, 1977, 1992-1995, 2016-\>exit, 2053-2054, 2084, 2088, 2131-\>exit, 2142, 2170-2171, 2193-2201, 2217-\>2224, 2297, 2346, 2365, 2369-\>2375, 2371-2374, 2405-2406, 2421-2423 |
| src/async\_batch\_llm/budget.py                              |       13 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/cache\_groups.py                       |      221 |        3 |       88 |        3 |     98.06% |304, 419, 461 |
| src/async\_batch\_llm/call\_pool.py                          |       91 |        0 |       16 |        0 |    100.00% |           |
| src/async\_batch\_llm/callable\_strategy.py                  |      144 |        4 |       50 |        6 |     94.85% |57-\>exit, 70-\>exit, 89, 132, 173, 247 |
| src/async\_batch\_llm/categories.py                          |       94 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/classifiers/\_\_init\_\_.py            |        5 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/classifiers/gemini.py                  |       92 |        7 |       50 |        5 |     91.55% |27-\>30, 41, 55-\>52, 60-61, 115-116, 223, 231 |
| src/async\_batch\_llm/classifiers/openai.py                  |       71 |        5 |       40 |        1 |     94.59% |114, 146-147, 181-182 |
| src/async\_batch\_llm/classifiers/openrouter.py              |       37 |        2 |       16 |        0 |     96.23% |     85-86 |
| src/async\_batch\_llm/classifiers/pydantic\_ai.py            |       16 |        2 |        6 |        0 |     90.91% |     16-17 |
| src/async\_batch\_llm/core/\_\_init\_\_.py                   |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/config.py                         |      271 |        4 |      146 |        4 |     98.08% |153, 155, 280, 592 |
| src/async\_batch\_llm/core/protocols.py                      |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/factory.py                             |       66 |        3 |       32 |        3 |     91.84% |108-\>exit, 119-\>exit, 221-226 |
| src/async\_batch\_llm/gateway.py                             |       11 |        0 |        2 |        0 |    100.00% |           |
| src/async\_batch\_llm/llm\_strategies.py                     |      204 |       11 |       54 |        9 |     92.25% |36-37, 71-\>69, 73, 88-90, 335-\>exit, 347-\>exit, 434, 452, 465-\>468, 636-\>638, 641, 782-784 |
| src/async\_batch\_llm/middleware/\_\_init\_\_.py             |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/middleware/base.py                     |       11 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/models.py                              |     1210 |       97 |      542 |       78 |     89.10% |55-57, 60-61, 111-127, 393-\>396, 471-\>exit, 479, 482, 520, 527, 551, 728-\>731, 733, 739-\>exit, 764, 845, 848, 895-899, 926, 937, 963-964, 980-\>1014, 991, 993, 1029, 1045-\>1063, 1084-\>1089, 1380-\>1429, 1383-\>1386, 1387-\>1423, 1403-1404, 1409, 1430-\>1434, 1432-1433, 1444, 1498-\>1508, 1501-\>1508, 1503-1504, 1524, 1527, 1530-\>1538, 1785, 1794-1799, 1836, 1864-1866, 1904, 1917, 1934-\>1936, 1937-\>1939, 1940-\>1942, 1954-1957, 1959-1963, 1971, 1973, 2038-2039, 2223, 2226-\>2228, 2229-\>2231, 2238, 2243-\>2248, 2255-\>2257, 2282-\>2280, 2286-\>2284, 2290-\>2277, 2300-\>2277, 2305-\>2301, 2307-\>2301, 2310-\>2308, 2327, 2378-2379, 2469, 2492, 2502-2503, 2506, 2518-\>2527, 2522-2526, 2533-2534, 2547, 2623, 2632, 2636-2637, 2671, 2674, 2677-2680, 2808, 2828-\>2833, 2874-\>2876 |
| src/async\_batch\_llm/observers/\_\_init\_\_.py              |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/observers/base.py                      |       25 |        1 |        0 |        0 |     96.00% |        63 |
| src/async\_batch\_llm/observers/metrics.py                   |      143 |        5 |       72 |       10 |     92.09% |56, 73-\>80, 83-\>89, 90-\>exit, 115-\>109, 118-\>exit, 129, 135-136, 153, 176-\>exit |
| src/async\_batch\_llm/parallel.py                            |      453 |       22 |      122 |        8 |     94.78% |113-114, 117-118, 354, 365, 369, 377, 381, 385, 389, 393, 405, 438-\>440, 528-\>530, 537, 551, 559, 592-593, 636-\>638, 704, 709, 804, 987-\>992, 1039 |
| src/async\_batch\_llm/parsing.py                             |       63 |        0 |       18 |        0 |    100.00% |           |
| src/async\_batch\_llm/provider\_output.py                    |      102 |        0 |       28 |        0 |    100.00% |           |
| src/async\_batch\_llm/serialization.py                       |      250 |       39 |      102 |       21 |     82.39% |117, 131, 140-141, 223-224, 228, 253, 259, 269, 271, 351-352, 367, 377-378, 387, 389, 403, 474, 481, 488, 493, 522, 539, 575, 606-609, 661-662, 673-674, 680, 683-684, 696, 707 |
| src/async\_batch\_llm/single.py                              |       35 |        3 |        4 |        1 |     89.74% | 54-55, 69 |
| src/async\_batch\_llm/sqlite\_artifacts.py                   |      721 |       89 |      208 |       39 |     84.93% |88-89, 126-\>exit, 134, 139, 218, 226-227, 276-277, 284-286, 291-296, 319, 361-362, 421-422, 427, 431, 461, 528-531, 551, 583, 602-605, 608-609, 611-\>618, 614-615, 626, 655, 663, 677, 683, 689-690, 712, 714-\>exit, 728-\>730, 731, 737-\>739, 748, 756-762, 799-800, 811-\>810, 826-\>825, 840-841, 853, 863-865, 881-\>896, 910, 930, 947-950, 954-956, 1049-1051, 1067, 1134, 1141, 1144, 1148, 1159, 1190, 1286, 1352-1353, 1399-1400, 1408-1409, 1412-1413, 1417 |
| src/async\_batch\_llm/strategies/\_\_init\_\_.py             |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/strategies/errors.py                   |      234 |       18 |       84 |        3 |     93.40% |90-\>94, 131-132, 137-138, 144-155, 306, 551-552, 589 |
| src/async\_batch\_llm/strategies/rate\_limit.py              |       32 |        0 |        2 |        0 |    100.00% |           |
| src/async\_batch\_llm/streaming.py                           |      217 |        2 |       80 |        5 |     97.64% |81-\>exit, 109, 218-\>220, 274-\>277, 370, 376-\>exit |
| src/async\_batch\_llm/token\_estimation.py                   |       35 |        0 |        4 |        0 |    100.00% |           |
| src/async\_batch\_llm/token\_extractor.py                    |      136 |        3 |       56 |        5 |     95.83% |88-\>98, 101-\>107, 103-\>107, 146-147, 154, 206-\>208 |
| **TOTAL**                                                    | **9604** |  **557** | **3196** |  **357** | **92.42%** |           |


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