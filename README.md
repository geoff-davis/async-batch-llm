# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/geoff-davis/async-batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                         |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|------------------------------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| src/async\_batch\_llm/\_\_init\_\_.py                        |       29 |        2 |        0 |        0 |     93.10% |   278-280 |
| src/async\_batch\_llm/\_internal/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/capacity.py                 |      117 |       11 |       28 |        2 |     89.66% |39-40, 74-78, 135-136, 156-162 |
| src/async\_batch\_llm/\_internal/error\_logging.py           |       76 |       18 |       28 |        9 |     72.12% |36-\>39, 87, 101-\>111, 109-110, 112, 115-126, 143-\>164, 145, 147-150, 159-162 |
| src/async\_batch\_llm/\_internal/event\_dispatcher.py        |       61 |        7 |       16 |        1 |     89.61% |70, 76-77, 94, 110, 124, 126 |
| src/async\_batch\_llm/\_internal/executor\_host.py           |       51 |        6 |       10 |        5 |     81.97% |50, 51-\>61, 53-\>61, 57-58, 60, 113-115 |
| src/async\_batch\_llm/\_internal/item\_executor.py           |      366 |       13 |       98 |       19 |     92.67% |166-\>exit, 168-\>exit, 177-\>exit, 271, 337, 360-\>366, 395, 440-\>443, 458-\>exit, 467-475, 602-\>exit, 686, 712-\>717, 726-\>733, 766-\>768, 789-\>794, 867-\>881, 871, 882-\>884, 889, 971, 978-\>981 |
| src/async\_batch\_llm/\_internal/rate\_limit\_coordinator.py |      117 |       13 |       24 |        2 |     89.36% |93, 99-101, 175, 233-245 |
| src/async\_batch\_llm/\_internal/strategy\_lifecycle.py      |       40 |        2 |       10 |        1 |     94.00% |79-\>78, 100, 104 |
| src/async\_batch\_llm/base.py                                |      453 |       43 |      118 |       13 |     88.79% |103, 266, 278, 282, 356-\>exit, 696, 698, 821-822, 851-852, 859-860, 864-874, 885-886, 959, 987, 1000, 1005-\>exit, 1039, 1043, 1094-1110, 1171, 1217, 1236, 1251-\>exit, 1260-1264, 1281-1283 |
| src/async\_batch\_llm/classifiers/\_\_init\_\_.py            |        4 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/classifiers/gemini.py                  |       59 |        6 |       28 |        2 |     90.80% |75-76, 110-111, 207, 215 |
| src/async\_batch\_llm/classifiers/openai.py                  |       62 |        7 |       32 |        3 |     89.36% |89-90, 98, 108, 174-175, 247 |
| src/async\_batch\_llm/classifiers/openrouter.py              |       19 |        0 |        8 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/\_\_init\_\_.py                   |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/config.py                         |      117 |        1 |       60 |        2 |     98.31% |278, 293-\>exit |
| src/async\_batch\_llm/core/protocols.py                      |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/gateway.py                             |       69 |        0 |       14 |        0 |    100.00% |           |
| src/async\_batch\_llm/llm\_strategies.py                     |      140 |        8 |       32 |        5 |     92.44% |31-32, 66-\>64, 68, 83-85, 308-\>exit, 318-\>exit, 630-632 |
| src/async\_batch\_llm/middleware/\_\_init\_\_.py             |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/middleware/base.py                     |       11 |        1 |        0 |        0 |     90.91% |        98 |
| src/async\_batch\_llm/models.py                              |      521 |       31 |      212 |       26 |     91.95% |40-42, 45-46, 256-\>259, 336-\>339, 344, 347, 367, 512-\>515, 517, 523-\>exit, 548, 627-\>630, 631, 634, 647-651, 687-688, 704-\>734, 715, 746, 780, 1047-\>1096, 1050-\>1053, 1054-\>1090, 1070-1071, 1076, 1097-\>1101, 1099-1100, 1107, 1121, 1124-\>exit, 1325-1326, 1515-\>1520 |
| src/async\_batch\_llm/observers/\_\_init\_\_.py              |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/observers/base.py                      |       20 |        1 |        0 |        0 |     95.00% |        58 |
| src/async\_batch\_llm/observers/metrics.py                   |       80 |        1 |       28 |        5 |     94.44% |36, 52-\>exit, 71-\>exit, 81-\>exit, 193-\>202 |
| src/async\_batch\_llm/parallel.py                            |      266 |       25 |       66 |        7 |     90.36% |71-72, 75-76, 140, 147, 180, 182, 286, 290, 301, 305, 309, 313, 317, 390-392, 462-\>465, 495, 500, 505, 576-\>583, 641-644, 678, 682-683 |
| src/async\_batch\_llm/parsing.py                             |       21 |        0 |        6 |        0 |    100.00% |           |
| src/async\_batch\_llm/provider\_output.py                    |       82 |        0 |       22 |        0 |    100.00% |           |
| src/async\_batch\_llm/single.py                              |       27 |        3 |        4 |        1 |     87.10% | 52-53, 67 |
| src/async\_batch\_llm/strategies/\_\_init\_\_.py             |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/strategies/errors.py                   |       95 |       12 |       20 |        1 |     88.70% |62-73, 130-\>exit, 329-330 |
| src/async\_batch\_llm/strategies/rate\_limit.py              |       31 |        0 |        2 |        0 |    100.00% |           |
| src/async\_batch\_llm/streaming.py                           |       55 |        0 |       20 |        0 |    100.00% |           |
| src/async\_batch\_llm/token\_extractor.py                    |       63 |        3 |       30 |        5 |     91.40% |51-\>61, 68-\>72, 76-\>89, 114, 132-\>135, 151-152 |
| **TOTAL**                                                    | **3065** |  **214** |  **916** |  **109** | **91.48%** |           |


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