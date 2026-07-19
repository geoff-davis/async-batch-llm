# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/geoff-davis/async-batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                         |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|------------------------------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| src/async\_batch\_llm/\_\_init\_\_.py                        |       32 |        2 |        0 |        0 |     93.75% |   322-324 |
| src/async\_batch\_llm/\_internal/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/capacity.py                 |      118 |       11 |       28 |        2 |     89.73% |42-43, 83-92, 158-159, 186-192 |
| src/async\_batch\_llm/\_internal/error\_logging.py           |       76 |       18 |       28 |        9 |     72.12% |36-\>39, 87, 101-\>111, 109-110, 112, 115-126, 143-\>164, 145, 147-150, 159-162 |
| src/async\_batch\_llm/\_internal/event\_dispatcher.py        |       61 |        6 |       16 |        1 |     90.91% |70, 76-77, 110, 124, 126 |
| src/async\_batch\_llm/\_internal/executor\_host.py           |       55 |        6 |       10 |        5 |     83.08% |51, 52-\>62, 54-\>62, 58-59, 61, 117-119 |
| src/async\_batch\_llm/\_internal/guardrails.py               |      128 |        4 |       40 |        5 |     94.64% |59, 69, 98, 136-\>exit, 154 |
| src/async\_batch\_llm/\_internal/item\_executor.py           |      439 |       19 |      122 |       28 |     91.62% |175-\>exit, 177-\>exit, 186-\>exit, 194-\>exit, 291, 309-\>314, 328-\>exit, 399, 418, 425-\>435, 462, 472-485, 500, 520-525, 558-\>561, 576-\>exit, 586, 717, 740-\>exit, 833, 869-\>874, 888-\>896, 936-\>938, 943-\>945, 966-\>971, 1051-\>1079, 1060, 1067-\>1069, 1080-\>1082, 1087, 1098-\>1100, 1178, 1185-\>1188 |
| src/async\_batch\_llm/\_internal/rate\_limit\_coordinator.py |      117 |       10 |       24 |        2 |     91.49% |93, 99-101, 175, 237-245 |
| src/async\_batch\_llm/\_internal/strategy\_lifecycle.py      |       40 |        2 |       10 |        1 |     94.00% |79-\>78, 100, 104 |
| src/async\_batch\_llm/artifacts.py                           |      400 |       56 |      116 |       26 |     83.33% |103-\>105, 113, 115-117, 178-179, 202-203, 205, 221-222, 229, 233-234, 253, 260, 266, 270, 275, 332-333, 335, 366, 377-\>399, 379-386, 396-397, 433-434, 470, 478, 534-539, 582, 593, 599-\>598, 625, 628, 664-665, 674, 688, 699, 704-\>706, 708, 715-716, 734-735, 741, 767-768 |
| src/async\_batch\_llm/base.py                                |      635 |       76 |      168 |       17 |     85.43% |107, 286, 290, 367-\>exit, 651-\>exit, 654-\>exit, 705, 742-\>744, 1015, 1017, 1143-1144, 1173-1174, 1181-1182, 1198-1208, 1221-1222, 1270-1272, 1315, 1344, 1357, 1361-1366, 1370-\>exit, 1404, 1408, 1444-1452, 1464, 1466-1483, 1501-1517, 1585, 1631, 1650, 1665-\>exit, 1674-1678, 1695-1697 |
| src/async\_batch\_llm/classifiers/\_\_init\_\_.py            |        4 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/classifiers/gemini.py                  |       69 |        9 |       38 |        5 |     86.92% |51, 53, 55, 84-85, 119-120, 222, 230 |
| src/async\_batch\_llm/classifiers/openai.py                  |       72 |       10 |       42 |        6 |     85.96% |68, 70, 72, 98-99, 107, 117, 183-184, 272 |
| src/async\_batch\_llm/classifiers/openrouter.py              |       19 |        0 |        8 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/\_\_init\_\_.py                   |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/config.py                         |      174 |        2 |       86 |        3 |     98.08% |200, 389, 405-\>exit |
| src/async\_batch\_llm/core/protocols.py                      |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/factory.py                             |       45 |        0 |       22 |        2 |     97.01% |63-\>exit, 74-\>exit |
| src/async\_batch\_llm/gateway.py                             |       70 |        0 |       14 |        0 |    100.00% |           |
| src/async\_batch\_llm/llm\_strategies.py                     |      146 |        9 |       34 |        6 |     91.67% |31-32, 66-\>64, 68, 83-85, 308-\>exit, 318-\>exit, 386, 644-646 |
| src/async\_batch\_llm/middleware/\_\_init\_\_.py             |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/middleware/base.py                     |       11 |        1 |        0 |        0 |     90.91% |        98 |
| src/async\_batch\_llm/models.py                              |      543 |       33 |      222 |       28 |     91.76% |40-42, 45-46, 256-\>259, 336-\>339, 344, 347, 367, 512-\>515, 517, 523-\>exit, 548, 627-\>630, 631, 634, 647-651, 687-688, 704-\>734, 715, 746, 780, 1052-\>1101, 1055-\>1058, 1059-\>1095, 1075-1076, 1081, 1102-\>1106, 1104-1105, 1112, 1156-\>1166, 1159-\>1166, 1161-1162, 1180, 1183-\>exit, 1387-1388, 1577-\>1582 |
| src/async\_batch\_llm/observers/\_\_init\_\_.py              |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/observers/base.py                      |       23 |        1 |        0 |        0 |     95.65% |        61 |
| src/async\_batch\_llm/observers/metrics.py                   |      107 |        8 |       46 |        9 |     87.58% |42, 58-\>65, 68-\>74, 75-\>exit, 91, 92-\>exit, 99-104, 107, 113-\>exit |
| src/async\_batch\_llm/parallel.py                            |      372 |       37 |      108 |       12 |     89.79% |79-80, 83-84, 150, 190, 305, 309, 320, 324, 328, 332, 336, 375, 391-392, 426-428, 480-481, 496, 509-512, 522-524, 596-\>599, 640, 645, 650, 742, 752, 757, 809-\>814, 838-\>843, 890, 894-895 |
| src/async\_batch\_llm/parsing.py                             |       63 |        0 |       18 |        0 |    100.00% |           |
| src/async\_batch\_llm/provider\_output.py                    |      102 |        1 |       28 |        1 |     98.46% |       225 |
| src/async\_batch\_llm/serialization.py                       |      239 |       39 |       96 |       21 |     81.49% |113, 116, 130, 139-140, 170, 222-223, 227, 252, 258, 268, 270, 313-314, 329, 339-340, 351, 422, 429, 436, 441, 459, 476, 512, 543-546, 598-599, 610-611, 617, 620-621, 633, 644 |
| src/async\_batch\_llm/single.py                              |       27 |        3 |        4 |        1 |     87.10% | 52-53, 67 |
| src/async\_batch\_llm/strategies/\_\_init\_\_.py             |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/strategies/errors.py                   |      113 |       12 |       26 |        1 |     90.65% |62-73, 130-\>exit, 375-376 |
| src/async\_batch\_llm/strategies/rate\_limit.py              |       31 |        0 |        2 |        0 |    100.00% |           |
| src/async\_batch\_llm/streaming.py                           |      154 |        1 |       64 |        3 |     98.17% |62-\>exit, 205-\>208, 261, 267-\>exit |
| src/async\_batch\_llm/token\_extractor.py                    |       63 |        3 |       30 |        5 |     91.40% |51-\>61, 68-\>72, 76-\>89, 114, 132-\>135, 151-152 |
| **TOTAL**                                                    | **4561** |  **379** | **1450** |  **199** | **89.75%** |           |


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