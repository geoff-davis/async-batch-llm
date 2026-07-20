# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/geoff-davis/async-batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                         |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|------------------------------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| src/async\_batch\_llm/\_\_init\_\_.py                        |       33 |        2 |        0 |        0 |     93.94% |   326-328 |
| src/async\_batch\_llm/\_internal/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/capacity.py                 |      118 |       11 |       28 |        2 |     89.73% |42-43, 83-92, 158-159, 186-192 |
| src/async\_batch\_llm/\_internal/error\_logging.py           |       76 |       18 |       28 |        9 |     72.12% |36-\>39, 87, 101-\>111, 109-110, 112, 115-126, 143-\>164, 145, 147-150, 159-162 |
| src/async\_batch\_llm/\_internal/event\_dispatcher.py        |       61 |        6 |       16 |        1 |     90.91% |70, 76-77, 110, 124, 126 |
| src/async\_batch\_llm/\_internal/executor\_host.py           |       56 |        4 |       10 |        3 |     89.39% |52-\>62, 54-\>62, 58-59, 117-119 |
| src/async\_batch\_llm/\_internal/guardrails.py               |      128 |        4 |       40 |        5 |     94.64% |59, 69, 98, 136-\>exit, 154 |
| src/async\_batch\_llm/\_internal/item\_executor.py           |      439 |       18 |      122 |       28 |     91.80% |175-\>exit, 177-\>exit, 186-\>exit, 194-\>exit, 291, 309-\>314, 328-\>exit, 399, 418, 425-\>435, 462, 472-485, 520-525, 558-\>561, 576-\>exit, 586, 717, 740-\>exit, 833, 869-\>874, 888-\>896, 936-\>938, 943-\>945, 966-\>971, 1051-\>1079, 1060, 1067-\>1069, 1080-\>1082, 1087, 1098-\>1100, 1178, 1185-\>1188 |
| src/async\_batch\_llm/\_internal/rate\_limit\_coordinator.py |      123 |        8 |       24 |        2 |     93.20% |100, 106-108, 201, 262-268 |
| src/async\_batch\_llm/\_internal/strategy\_lifecycle.py      |       40 |        2 |       10 |        1 |     94.00% |79-\>78, 100, 104 |
| src/async\_batch\_llm/artifacts.py                           |      408 |       57 |      122 |       27 |     83.40% |107, 118-\>120, 128, 130-132, 193-194, 217-218, 220, 236-237, 244, 248-249, 268, 275, 281, 285, 290, 347-348, 350, 381, 392-\>414, 394-401, 411-412, 448-449, 485, 493, 549-554, 597, 608, 614-\>613, 640, 643, 679-680, 689, 703, 714, 719-\>721, 723, 730-731, 749-750, 756, 782-783 |
| src/async\_batch\_llm/base.py                                |      679 |       78 |      186 |       19 |     85.78% |107, 286, 290, 367-\>exit, 651-\>exit, 654-\>exit, 705, 742-\>744, 1015, 1017, 1152-1153, 1182-1183, 1194-1195, 1211-1221, 1234-1235, 1328, 1363, 1380, 1384-1389, 1393-\>exit, 1399, 1409-1412, 1416-\>exit, 1477, 1481, 1517-1525, 1537, 1539-1556, 1574-1590, 1658, 1704, 1723, 1738-\>exit, 1747-1751, 1768-1770 |
| src/async\_batch\_llm/callable\_strategy.py                  |      120 |        3 |       44 |        6 |     94.51% |49-\>exit, 62-\>exit, 81, 121, 160, 231-\>237 |
| src/async\_batch\_llm/classifiers/\_\_init\_\_.py            |        4 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/classifiers/gemini.py                  |       69 |        9 |       38 |        5 |     86.92% |51, 53, 55, 84-85, 119-120, 222, 230 |
| src/async\_batch\_llm/classifiers/openai.py                  |       72 |       10 |       42 |        6 |     85.96% |68, 70, 72, 98-99, 107, 117, 183-184, 272 |
| src/async\_batch\_llm/classifiers/openrouter.py              |       19 |        0 |        8 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/\_\_init\_\_.py                   |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/config.py                         |      177 |        2 |       88 |        3 |     98.11% |200, 405, 421-\>exit |
| src/async\_batch\_llm/core/protocols.py                      |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/factory.py                             |       45 |        0 |       22 |        2 |     97.01% |63-\>exit, 74-\>exit |
| src/async\_batch\_llm/gateway.py                             |       72 |        0 |       14 |        0 |    100.00% |           |
| src/async\_batch\_llm/llm\_strategies.py                     |      146 |        9 |       34 |        6 |     91.67% |31-32, 66-\>64, 68, 83-85, 299-\>exit, 309-\>exit, 377, 635-637 |
| src/async\_batch\_llm/middleware/\_\_init\_\_.py             |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/middleware/base.py                     |       11 |        1 |        0 |        0 |     90.91% |        98 |
| src/async\_batch\_llm/models.py                              |      543 |       33 |      222 |       28 |     91.76% |40-42, 45-46, 256-\>259, 336-\>339, 344, 347, 367, 512-\>515, 517, 523-\>exit, 548, 627-\>630, 631, 634, 647-651, 687-688, 704-\>734, 715, 746, 780, 1052-\>1101, 1055-\>1058, 1059-\>1095, 1075-1076, 1081, 1102-\>1106, 1104-1105, 1112, 1156-\>1166, 1159-\>1166, 1161-1162, 1180, 1183-\>exit, 1387-1388, 1577-\>1582 |
| src/async\_batch\_llm/observers/\_\_init\_\_.py              |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/observers/base.py                      |       23 |        1 |        0 |        0 |     95.65% |        61 |
| src/async\_batch\_llm/observers/metrics.py                   |      107 |        8 |       46 |        9 |     87.58% |42, 58-\>65, 68-\>74, 75-\>exit, 91, 92-\>exit, 99-104, 107, 113-\>exit |
| src/async\_batch\_llm/parallel.py                            |      373 |       32 |      108 |       11 |     91.06% |79-80, 83-84, 150, 190, 306, 310, 321, 325, 329, 333, 337, 376, 484-485, 500, 513-516, 526-528, 600-\>603, 644, 649, 654, 746, 756, 761, 813-\>818, 842-\>847, 894, 898-899 |
| src/async\_batch\_llm/parsing.py                             |       63 |        0 |       18 |        0 |    100.00% |           |
| src/async\_batch\_llm/provider\_output.py                    |      102 |        1 |       28 |        1 |     98.46% |       225 |
| src/async\_batch\_llm/serialization.py                       |      239 |       39 |       96 |       21 |     81.49% |113, 116, 130, 139-140, 170, 222-223, 227, 252, 258, 268, 270, 313-314, 329, 339-340, 351, 422, 429, 436, 441, 459, 476, 512, 543-546, 598-599, 610-611, 617, 620-621, 633, 644 |
| src/async\_batch\_llm/single.py                              |       27 |        3 |        4 |        1 |     87.10% | 52-53, 67 |
| src/async\_batch\_llm/strategies/\_\_init\_\_.py             |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/strategies/errors.py                   |      113 |       12 |       26 |        1 |     90.65% |62-73, 130-\>exit, 375-376 |
| src/async\_batch\_llm/strategies/rate\_limit.py              |       31 |        0 |        2 |        0 |    100.00% |           |
| src/async\_batch\_llm/streaming.py                           |      154 |        1 |       64 |        3 |     98.17% |62-\>exit, 205-\>208, 263, 269-\>exit |
| src/async\_batch\_llm/token\_extractor.py                    |       63 |        3 |       30 |        5 |     91.40% |51-\>61, 68-\>72, 76-\>89, 114, 132-\>135, 151-152 |
| **TOTAL**                                                    | **4747** |  **375** | **1520** |  **205** | **90.11%** |           |


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