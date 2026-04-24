# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/geoff-davis/async-batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                         |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|------------------------------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| src/async\_batch\_llm/\_\_init\_\_.py                        |       24 |        2 |        0 |        0 |     91.67% |   207-209 |
| src/async\_batch\_llm/\_internal/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/error\_logging.py           |       76 |       31 |       28 |        7 |     53.85% |36-\>39, 90, 95-129, 146-\>167, 148, 150-153, 159-165 |
| src/async\_batch\_llm/\_internal/event\_dispatcher.py        |       59 |       16 |       14 |        1 |     76.71% |57, 63-64, 80-83, 96-99, 111-115 |
| src/async\_batch\_llm/\_internal/rate\_limit\_coordinator.py |      109 |       12 |       18 |        1 |     89.76% |86-88, 149, 187-199 |
| src/async\_batch\_llm/\_internal/strategy\_lifecycle.py      |       40 |        2 |       10 |        1 |     94.00% |82-\>81, 105, 109 |
| src/async\_batch\_llm/base.py                                |      229 |       36 |       48 |        7 |     81.59% |90, 401-402, 416, 424-425, 429-433, 437-447, 470, 474, 521-537, 589-\>exit, 594, 611, 626-\>exit, 635-639, 656-658 |
| src/async\_batch\_llm/classifiers/\_\_init\_\_.py            |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/classifiers/gemini.py                  |       44 |       12 |       16 |        2 |     76.67% |34-36, 44-46, 55-57, 75-76, 111-112 |
| src/async\_batch\_llm/core/\_\_init\_\_.py                   |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/config.py                         |       86 |        1 |       42 |        2 |     97.66% |164, 201-\>exit |
| src/async\_batch\_llm/core/protocols.py                      |       15 |        4 |        0 |        0 |     73.33% |22, 31, 35, 39 |
| src/async\_batch\_llm/llm\_strategies.py                     |       82 |       10 |       16 |        5 |     84.69% |28-29, 210-\>exit, 219-\>exit, 258-\>exit, 283-285, 350-352, 376-378 |
| src/async\_batch\_llm/middleware/\_\_init\_\_.py             |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/middleware/base.py                     |       11 |        1 |        0 |        0 |     90.91% |        82 |
| src/async\_batch\_llm/models.py                              |      216 |       18 |       84 |       14 |     88.67% |31-33, 116-\>119, 191, 194, 213, 347-\>350, 352, 358-\>exit, 383, 438, 468, 471, 483-487, 516, 540 |
| src/async\_batch\_llm/observers/\_\_init\_\_.py              |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/observers/base.py                      |       18 |        1 |        0 |        0 |     94.44% |        51 |
| src/async\_batch\_llm/observers/metrics.py                   |       71 |        1 |       26 |        5 |     93.81% |16, 42-\>exit, 53-\>exit, 63-\>exit, 169-\>178 |
| src/async\_batch\_llm/parallel.py                            |      353 |       35 |       98 |       19 |     87.58% |93, 100, 107, 126, 128, 130, 205, 209, 220, 224, 228, 232, 236, 303-\>306, 346-348, 368, 392, 419-\>426, 457-458, 479-482, 511, 551, 574-578, 592-\>594, 625-\>634, 652, 707, 722-724, 813-\>825, 817, 894, 901-\>904, 942 |
| src/async\_batch\_llm/strategies/\_\_init\_\_.py             |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/strategies/errors.py                   |       48 |        2 |       12 |        0 |     96.67% |   177-178 |
| src/async\_batch\_llm/strategies/rate\_limit.py              |       31 |        0 |        2 |        0 |    100.00% |           |
| src/async\_batch\_llm/token\_extractor.py                    |       47 |        2 |       20 |        4 |     91.04% |51-\>57, 53-\>57, 61-\>65, 66-\>83, 110-111 |
| **TOTAL**                                                    | **1572** |  **186** |  **434** |   **68** | **86.24%** |           |


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