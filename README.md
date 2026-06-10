# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/geoff-davis/async-batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                         |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|------------------------------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| src/async\_batch\_llm/\_\_init\_\_.py                        |       25 |        2 |        0 |        0 |     92.00% |   240-242 |
| src/async\_batch\_llm/\_internal/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/error\_logging.py           |       76 |       18 |       28 |        9 |     72.12% |36-\>39, 87, 101-\>111, 109-110, 112, 115-126, 143-\>164, 145, 147-150, 159-162 |
| src/async\_batch\_llm/\_internal/event\_dispatcher.py        |       59 |       16 |       14 |        1 |     76.71% |57, 63-64, 80-83, 96-99, 111-115 |
| src/async\_batch\_llm/\_internal/rate\_limit\_coordinator.py |      112 |       12 |       20 |        1 |     90.15% |84-86, 160, 209-221 |
| src/async\_batch\_llm/\_internal/strategy\_lifecycle.py      |       40 |        2 |       10 |        1 |     94.00% |79-\>78, 100, 104 |
| src/async\_batch\_llm/base.py                                |      265 |       34 |       66 |        8 |     84.89% |90, 262-\>exit, 557-558, 572, 580-581, 588-589, 593-603, 645, 649, 696-712, 772-\>exit, 776, 778, 795, 810-\>exit, 819-823, 840-842 |
| src/async\_batch\_llm/classifiers/\_\_init\_\_.py            |        4 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/classifiers/gemini.py                  |       45 |        9 |       18 |        1 |     84.13% |46-48, 56-58, 100-101, 135-136 |
| src/async\_batch\_llm/classifiers/openai.py                  |       82 |       17 |       36 |        3 |     83.05% |59-70, 119-120, 128, 138, 204-205, 277 |
| src/async\_batch\_llm/classifiers/openrouter.py              |       10 |        0 |        2 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/\_\_init\_\_.py                   |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/config.py                         |       86 |        1 |       42 |        2 |     97.66% |177, 214-\>exit |
| src/async\_batch\_llm/core/protocols.py                      |       15 |        4 |        0 |        0 |     73.33% |22, 31, 35, 39 |
| src/async\_batch\_llm/llm\_strategies.py                     |       92 |        8 |       22 |        5 |     88.60% |28-29, 48-\>46, 50, 65-67, 252-\>exit, 261-\>exit, 506-508 |
| src/async\_batch\_llm/middleware/\_\_init\_\_.py             |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/middleware/base.py                     |       11 |        1 |        0 |        0 |     90.91% |        82 |
| src/async\_batch\_llm/models.py                              |      415 |       26 |      168 |       23 |     91.25% |32-34, 37-38, 125-\>128, 195-\>198, 203, 206, 225, 359-\>362, 364, 370-\>exit, 395, 474-\>477, 478, 481, 493-497, 532-533, 549-\>579, 560, 591, 625, 841-\>845, 843-\>845, 846-\>850, 848-849, 856, 870, 873-\>exit, 1228-\>1233 |
| src/async\_batch\_llm/observers/\_\_init\_\_.py              |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/observers/base.py                      |       18 |        1 |        0 |        0 |     94.44% |        51 |
| src/async\_batch\_llm/observers/metrics.py                   |       71 |        1 |       26 |        5 |     93.81% |16, 42-\>exit, 53-\>exit, 63-\>exit, 169-\>178 |
| src/async\_batch\_llm/parallel.py                            |      388 |       42 |      110 |       20 |     86.75% |73-74, 77-78, 142, 149, 156, 182, 184, 186, 190, 268, 272, 283, 287, 291, 295, 299, 371-\>374, 414-416, 436, 460, 475-\>483, 498-\>505, 554-557, 591, 631, 657-665, 679-\>681, 747, 820, 835-839, 936-\>948, 940, 1019, 1026-\>1029, 1067 |
| src/async\_batch\_llm/parsing.py                             |       21 |        0 |        6 |        0 |    100.00% |           |
| src/async\_batch\_llm/strategies/\_\_init\_\_.py             |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/strategies/errors.py                   |       59 |        2 |       14 |        0 |     97.26% |   224-225 |
| src/async\_batch\_llm/strategies/rate\_limit.py              |       31 |        0 |        2 |        0 |    100.00% |           |
| src/async\_batch\_llm/token\_extractor.py                    |       60 |        3 |       28 |        6 |     89.77% |51-\>57, 53-\>57, 61-\>65, 66-\>83, 108, 126-\>131, 142-143 |
| **TOTAL**                                                    | **1996** |  **199** |  **612** |   **85** | **88.50%** |           |


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