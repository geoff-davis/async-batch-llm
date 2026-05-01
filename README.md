# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/geoff-davis/async-batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                         |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|------------------------------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| src/async\_batch\_llm/\_\_init\_\_.py                        |       24 |        2 |        0 |        0 |     91.67% |   228-230 |
| src/async\_batch\_llm/\_internal/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/error\_logging.py           |       76 |       31 |       28 |        7 |     53.85% |36-\>39, 87, 92-126, 143-\>164, 145, 147-150, 156-162 |
| src/async\_batch\_llm/\_internal/event\_dispatcher.py        |       59 |       16 |       14 |        1 |     76.71% |57, 63-64, 80-83, 96-99, 111-115 |
| src/async\_batch\_llm/\_internal/rate\_limit\_coordinator.py |      109 |       12 |       18 |        1 |     89.76% |84-86, 147, 185-197 |
| src/async\_batch\_llm/\_internal/strategy\_lifecycle.py      |       40 |        2 |       10 |        1 |     94.00% |79-\>78, 100, 104 |
| src/async\_batch\_llm/base.py                                |      256 |       36 |       60 |        8 |     83.54% |90, 263-\>exit, 535-536, 550, 558-559, 563-567, 571-581, 604, 608, 655-671, 723-\>exit, 728, 745, 760-\>exit, 769-773, 790-792 |
| src/async\_batch\_llm/classifiers/\_\_init\_\_.py            |        4 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/classifiers/gemini.py                  |       44 |       12 |       16 |        2 |     76.67% |34-36, 44-46, 55-57, 75-76, 111-112 |
| src/async\_batch\_llm/classifiers/openai.py                  |       58 |        7 |       28 |        3 |     88.37% |73-74, 82, 92, 146-147, 209 |
| src/async\_batch\_llm/classifiers/openrouter.py              |       10 |        0 |        2 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/\_\_init\_\_.py                   |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/config.py                         |       86 |        1 |       42 |        2 |     97.66% |164, 201-\>exit |
| src/async\_batch\_llm/core/protocols.py                      |       15 |        4 |        0 |        0 |     73.33% |22, 31, 35, 39 |
| src/async\_batch\_llm/llm\_strategies.py                     |      138 |       26 |       36 |       11 |     75.29% |28-29, 221-\>exit, 230-\>exit, 269-\>exit, 298-300, 330-\>exit, 339-\>exit, 369, 392-394, 421-\>exit, 430-\>exit, 457-458, 462-463, 480-488, 555-557, 581-583 |
| src/async\_batch\_llm/middleware/\_\_init\_\_.py             |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/middleware/base.py                     |       11 |        1 |        0 |        0 |     90.91% |        82 |
| src/async\_batch\_llm/models.py                              |      346 |       24 |      138 |       19 |     90.70% |32-34, 37-38, 121-\>124, 196, 199, 218, 352-\>355, 357, 363-\>exit, 388, 443, 473, 476, 488-492, 521, 545, 755-\>759, 757-\>759, 760-\>764, 762-763, 770, 784, 787-\>exit |
| src/async\_batch\_llm/observers/\_\_init\_\_.py              |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/observers/base.py                      |       18 |        1 |        0 |        0 |     94.44% |        51 |
| src/async\_batch\_llm/observers/metrics.py                   |       71 |        1 |       26 |        5 |     93.81% |16, 42-\>exit, 53-\>exit, 63-\>exit, 169-\>178 |
| src/async\_batch\_llm/parallel.py                            |      355 |       36 |       98 |       19 |     87.42% |94, 101, 108, 127, 129, 131, 206, 210, 221, 225, 229, 233, 237, 304-\>307, 347-349, 369, 393, 420-\>427, 458-459, 480-483, 512, 552, 575-579, 593-\>595, 626-\>635, 653, 708, 723-727, 820-\>832, 824, 901, 908-\>911, 949 |
| src/async\_batch\_llm/strategies/\_\_init\_\_.py             |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/strategies/errors.py                   |       48 |        2 |       12 |        0 |     96.67% |   177-178 |
| src/async\_batch\_llm/strategies/rate\_limit.py              |       31 |        0 |        2 |        0 |    100.00% |           |
| src/async\_batch\_llm/token\_extractor.py                    |       54 |        2 |       24 |        5 |     91.03% |51-\>57, 53-\>57, 61-\>65, 66-\>83, 119-\>124, 135-136 |
| **TOTAL**                                                    | **1864** |  **216** |  **554** |   **84** | **86.44%** |           |


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