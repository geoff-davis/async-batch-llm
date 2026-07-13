# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/geoff-davis/async-batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                         |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|------------------------------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| src/async\_batch\_llm/\_\_init\_\_.py                        |       31 |        2 |        0 |        0 |     93.55% |   317-319 |
| src/async\_batch\_llm/\_internal/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/capacity.py                 |      118 |       11 |       28 |        2 |     89.73% |42-43, 83-92, 158-159, 186-192 |
| src/async\_batch\_llm/\_internal/error\_logging.py           |       76 |       18 |       28 |        9 |     72.12% |36-\>39, 87, 101-\>111, 109-110, 112, 115-126, 143-\>164, 145, 147-150, 159-162 |
| src/async\_batch\_llm/\_internal/event\_dispatcher.py        |       61 |        6 |       16 |        1 |     90.91% |70, 76-77, 110, 124, 126 |
| src/async\_batch\_llm/\_internal/executor\_host.py           |       55 |        6 |       10 |        5 |     83.08% |51, 52-\>62, 54-\>62, 58-59, 61, 117-119 |
| src/async\_batch\_llm/\_internal/guardrails.py               |      128 |        4 |       40 |        5 |     94.64% |59, 69, 98, 136-\>exit, 154 |
| src/async\_batch\_llm/\_internal/item\_executor.py           |      438 |       19 |      122 |       28 |     91.61% |175-\>exit, 177-\>exit, 186-\>exit, 194-\>exit, 291, 309-\>314, 328-\>exit, 399, 418, 425-\>435, 462, 472-485, 500, 520-525, 558-\>561, 576-\>exit, 586, 717, 740-\>exit, 831, 867-\>872, 886-\>894, 934-\>936, 941-\>943, 964-\>969, 1049-\>1077, 1058, 1065-\>1067, 1078-\>1080, 1085, 1096-\>1098, 1176, 1183-\>1186 |
| src/async\_batch\_llm/\_internal/rate\_limit\_coordinator.py |      117 |       10 |       24 |        2 |     91.49% |93, 99-101, 175, 237-245 |
| src/async\_batch\_llm/\_internal/strategy\_lifecycle.py      |       40 |        2 |       10 |        1 |     94.00% |79-\>78, 100, 104 |
| src/async\_batch\_llm/artifacts.py                           |      356 |       61 |       94 |       21 |     80.00% |116-117, 140-141, 143, 159-160, 167, 171-172, 191, 198, 204, 208, 213, 263-264, 266, 288-\>310, 290-297, 307-308, 344-345, 381, 389, 445-450, 490, 501, 507-\>506, 533, 536, 563-566, 575, 589, 600, 605-617, 631-632, 635-636, 642, 668-669 |
| src/async\_batch\_llm/base.py                                |      562 |       75 |      138 |       14 |     84.14% |106, 273, 285, 289, 366-\>exit, 848, 850, 976-977, 1006-1007, 1014-1015, 1019-1029, 1040-1041, 1089-1091, 1134, 1163, 1176, 1180-1185, 1189-\>exit, 1223, 1227, 1263-1271, 1283, 1285-1302, 1320-1336, 1397, 1443, 1462, 1477-\>exit, 1486-1490, 1507-1509 |
| src/async\_batch\_llm/classifiers/\_\_init\_\_.py            |        4 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/classifiers/gemini.py                  |       69 |        9 |       38 |        5 |     86.92% |51, 53, 55, 84-85, 119-120, 222, 230 |
| src/async\_batch\_llm/classifiers/openai.py                  |       72 |       10 |       42 |        6 |     85.96% |68, 70, 72, 98-99, 107, 117, 183-184, 272 |
| src/async\_batch\_llm/classifiers/openrouter.py              |       19 |        0 |        8 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/\_\_init\_\_.py                   |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/config.py                         |      139 |        2 |       66 |        3 |     97.56% |197, 319, 334-\>exit |
| src/async\_batch\_llm/core/protocols.py                      |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/gateway.py                             |       69 |        0 |       14 |        0 |    100.00% |           |
| src/async\_batch\_llm/llm\_strategies.py                     |      140 |        8 |       32 |        5 |     92.44% |31-32, 66-\>64, 68, 83-85, 308-\>exit, 318-\>exit, 630-632 |
| src/async\_batch\_llm/middleware/\_\_init\_\_.py             |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/middleware/base.py                     |       11 |        1 |        0 |        0 |     90.91% |        98 |
| src/async\_batch\_llm/models.py                              |      521 |       31 |      212 |       26 |     91.95% |40-42, 45-46, 256-\>259, 336-\>339, 344, 347, 367, 512-\>515, 517, 523-\>exit, 548, 627-\>630, 631, 634, 647-651, 687-688, 704-\>734, 715, 746, 780, 1047-\>1096, 1050-\>1053, 1054-\>1090, 1070-1071, 1076, 1097-\>1101, 1099-1100, 1107, 1121, 1124-\>exit, 1325-1326, 1515-\>1520 |
| src/async\_batch\_llm/observers/\_\_init\_\_.py              |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/observers/base.py                      |       23 |        1 |        0 |        0 |     95.65% |        61 |
| src/async\_batch\_llm/observers/metrics.py                   |      107 |        8 |       46 |        9 |     87.58% |42, 58-\>65, 68-\>74, 75-\>exit, 91, 92-\>exit, 99-104, 107, 113-\>exit |
| src/async\_batch\_llm/parallel.py                            |      366 |       37 |      106 |       14 |     89.19% |79-80, 83-84, 150, 157, 190, 192, 303, 307, 318, 322, 326, 330, 334, 373, 389-390, 424-426, 476, 489-492, 502-504, 576-\>579, 620, 625, 630, 722, 732, 737, 783-\>788, 812-\>817, 865, 869-870 |
| src/async\_batch\_llm/parsing.py                             |       63 |        0 |       18 |        0 |    100.00% |           |
| src/async\_batch\_llm/provider\_output.py                    |      102 |        1 |       28 |        1 |     98.46% |       225 |
| src/async\_batch\_llm/serialization.py                       |      229 |       38 |       90 |       20 |     81.19% |113, 116, 130, 139-140, 170, 222-223, 227, 252, 258, 268, 270, 313-314, 329, 339-340, 414, 421, 428, 433, 451, 468, 503, 531-534, 584-585, 596-597, 602, 605-606, 613, 624 |
| src/async\_batch\_llm/single.py                              |       27 |        3 |        4 |        1 |     87.10% | 52-53, 67 |
| src/async\_batch\_llm/strategies/\_\_init\_\_.py             |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/strategies/errors.py                   |      113 |       12 |       26 |        1 |     90.65% |62-73, 130-\>exit, 375-376 |
| src/async\_batch\_llm/strategies/rate\_limit.py              |       31 |        0 |        2 |        0 |    100.00% |           |
| src/async\_batch\_llm/streaming.py                           |       79 |        1 |       32 |        2 |     97.30% |55-\>exit, 131, 137-\>exit |
| src/async\_batch\_llm/token\_extractor.py                    |       63 |        3 |       30 |        5 |     91.40% |51-\>61, 68-\>72, 76-\>89, 114, 132-\>135, 151-152 |
| **TOTAL**                                                    | **4242** |  **379** | **1304** |  **186** | **89.09%** |           |


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