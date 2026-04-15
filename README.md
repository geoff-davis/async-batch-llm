# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/geoff-davis/async-batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                              |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|-------------------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| src/async\_batch\_llm/\_\_init\_\_.py             |       24 |        2 |        0 |        0 |     91.67% |   207-209 |
| src/async\_batch\_llm/base.py                     |      225 |       38 |       44 |        9 |     79.55% |90, 195, 200, 390-391, 405, 413-414, 418-422, 426-436, 459, 463, 508-524, 577-\>exit, 582, 599, 614-\>exit, 623-627, 644-646 |
| src/async\_batch\_llm/classifiers/\_\_init\_\_.py |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/classifiers/gemini.py       |       44 |       12 |       16 |        2 |     76.67% |34-36, 44-46, 55-57, 75-76, 111-112 |
| src/async\_batch\_llm/core/\_\_init\_\_.py        |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/config.py              |       83 |        1 |       40 |        2 |     97.56% |156, 193-\>exit |
| src/async\_batch\_llm/core/protocols.py           |       15 |        4 |        0 |        0 |     73.33% |22, 31, 35, 39 |
| src/async\_batch\_llm/llm\_strategies.py          |       78 |       10 |       12 |        3 |     85.56% |28-29, 234-\>exit, 259-261, 326-328, 352-354 |
| src/async\_batch\_llm/middleware/\_\_init\_\_.py  |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/middleware/base.py          |       11 |        1 |        0 |        0 |     90.91% |        82 |
| src/async\_batch\_llm/models.py                   |      214 |       36 |       82 |       17 |     79.39% |24-26, 75-\>78, 150, 153, 172, 289-\>292, 294, 300-\>exit, 308-\>exit, 359, 379, 382, 394-398, 427, 445, 468-471, 480-504, 529-531 |
| src/async\_batch\_llm/observers/\_\_init\_\_.py   |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/observers/base.py           |       18 |        1 |        0 |        0 |     94.44% |        51 |
| src/async\_batch\_llm/observers/metrics.py        |       71 |        1 |       26 |        5 |     93.81% |16, 42-\>exit, 53-\>exit, 63-\>exit, 169-\>178 |
| src/async\_batch\_llm/parallel.py                 |      545 |       94 |      186 |       35 |     80.16% |87, 94, 101, 120, 122, 124, 192, 196-\>195, 299-\>302, 328, 333-334, 347-350, 364-367, 378-382, 393-395, 421-422, 431, 455, 482-\>489, 519-520, 541-544, 610, 646-657, 712-\>722, 714-\>722, 723-727, 740-\>752, 743-747, 783, 802-\>816, 817-821, 835-\>837, 865-\>873, 892, 947, 962-964, 1054-\>1066, 1058, 1135, 1142-\>1145, 1189-\>1194, 1237-\>1264, 1240, 1242-1245, 1255-1261, 1285, 1289-1318 |
| src/async\_batch\_llm/strategies/\_\_init\_\_.py  |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/strategies/errors.py        |       48 |        2 |       12 |        0 |     96.67% |   177-178 |
| src/async\_batch\_llm/strategies/rate\_limit.py   |       31 |        0 |        2 |        0 |    100.00% |           |
| **TOTAL**                                         | **1420** |  **202** |  **420** |   **73** | **83.32%** |           |


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