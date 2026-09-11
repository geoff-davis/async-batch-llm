# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/geoff-davis/async-batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                         |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|------------------------------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| src/async\_batch\_llm/\_\_init\_\_.py                        |       36 |        2 |        0 |        0 |     94.44% |   355-357 |
| src/async\_batch\_llm/\_internal/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/admission.py                |      410 |       17 |      116 |       14 |     94.11% |102, 106, 114, 131, 133, 135, 155, 171, 173, 261, 293-294, 378, 392, 397-398, 509-\>507, 582 |
| src/async\_batch\_llm/\_internal/artifact\_codec.py          |      150 |       13 |       36 |        6 |     89.78% |56, 60, 64, 93, 127, 293, 319, 327, 372, 388-391 |
| src/async\_batch\_llm/\_internal/capacity.py                 |      131 |       11 |       32 |        2 |     90.80% |43-44, 84-93, 159-160, 187-193 |
| src/async\_batch\_llm/\_internal/classifier\_resolver.py     |       38 |        0 |        4 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/cleanup.py                  |      261 |        8 |       94 |        7 |     94.65% |142-\>exit, 148, 175, 295-\>297, 301-\>exit, 382, 415-\>428, 469-473 |
| src/async\_batch\_llm/\_internal/error\_logging.py           |       76 |       16 |       28 |        8 |     75.00% |36-\>39, 87, 101-\>111, 109-110, 112, 115-126, 143-\>164, 145, 147-150, 161-162 |
| src/async\_batch\_llm/\_internal/event\_dispatcher.py        |       81 |        5 |       22 |        2 |     93.20% |78-79, 118-\>120, 133, 147, 149 |
| src/async\_batch\_llm/\_internal/execution\_state.py         |       57 |        1 |        6 |        1 |     96.83% |       123 |
| src/async\_batch\_llm/\_internal/executor\_host.py           |       60 |        0 |        2 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/guardrails.py               |      136 |        9 |       44 |        5 |     91.11% |59, 69, 98, 137-\>exit, 155, 167-171 |
| src/async\_batch\_llm/\_internal/item\_executor.py           |      682 |       24 |      208 |       36 |     93.26% |208-\>exit, 210-\>exit, 214-\>exit, 223-\>exit, 231-\>exit, 259, 275, 344, 404, 450-\>exit, 459-\>461, 462-\>464, 478-481, 538, 564-\>572, 609-\>613, 627-\>exit, 671, 673, 807, 826, 833-\>843, 916, 991-998, 1037, 1151, 1241-1242, 1245-\>1250, 1310-\>1316, 1333-\>exit, 1395, 1411-\>1415, 1466-\>1470, 1475-\>1479, 1507-\>1512, 1518-\>1543, 1532-\>1543, 1543-\>1548, 1627-\>1634, 1683, 1685-\>1687, 1796 |
| src/async\_batch\_llm/\_internal/logical\_item.py            |       16 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/rate\_limit\_coordinator.py |      157 |        5 |       36 |        1 |     96.89% |117, 277, 349-355 |
| src/async\_batch\_llm/\_internal/strategy\_lifecycle.py      |       69 |        4 |       14 |        2 |     92.77% |66, 92, 100, 167 |
| src/async\_batch\_llm/artifacts.py                           |      462 |       72 |      150 |       33 |     82.19% |121, 132-\>134, 142, 144-146, 180-181, 242-243, 251-252, 259, 263-264, 283, 290, 296, 300, 305, 316, 318-321, 323, 346, 351, 366, 371-376, 382, 388, 405-406, 410, 466-467, 470-471, 473, 509, 527-528, 575-576, 601, 654-659, 668, 679, 685-\>684, 711, 714, 750-751, 760, 774, 785, 791-\>803, 796-797, 825, 832-835, 846-847, 856-857, 863 |
| src/async\_batch\_llm/base.py                                |      906 |       47 |      244 |       23 |     93.22% |119, 334, 343, 420-\>exit, 709-\>exit, 712-\>exit, 763, 800-\>802, 855-\>860, 1137, 1139, 1472, 1514-1515, 1671, 1709, 1761-1766, 1791, 1806-1809, 1830-\>exit, 1867-1868, 1898, 1902, 1941-\>exit, 1952, 1975, 1977-1981, 2003-2011, 2105, 2154, 2173, 2177-\>2183, 2179-2182, 2213-2214, 2229-2231 |
| src/async\_batch\_llm/callable\_strategy.py                  |      130 |        4 |       44 |        6 |     94.25% |51-\>exit, 64-\>exit, 83, 119, 160, 219 |
| src/async\_batch\_llm/classifiers/\_\_init\_\_.py            |        4 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/classifiers/gemini.py                  |       69 |        9 |       38 |        5 |     86.92% |51, 53, 55, 84-85, 119-120, 222, 230 |
| src/async\_batch\_llm/classifiers/openai.py                  |       76 |       10 |       46 |        6 |     86.89% |70, 72, 74, 116-117, 125, 135, 201-202, 290 |
| src/async\_batch\_llm/classifiers/openrouter.py              |       19 |        0 |        8 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/\_\_init\_\_.py                   |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/config.py                         |      184 |        3 |       92 |        3 |     97.83% |205, 440, 448 |
| src/async\_batch\_llm/core/protocols.py                      |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/factory.py                             |       45 |        0 |       22 |        2 |     97.01% |63-\>exit, 74-\>exit |
| src/async\_batch\_llm/gateway.py                             |       79 |        0 |       14 |        0 |    100.00% |           |
| src/async\_batch\_llm/llm\_strategies.py                     |      179 |       13 |       40 |        8 |     90.41% |33-34, 68-\>66, 70, 85-87, 279-280, 332-\>exit, 344-\>exit, 427, 445, 608-\>610, 613, 741-743 |
| src/async\_batch\_llm/middleware/\_\_init\_\_.py             |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/middleware/base.py                     |       11 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/models.py                              |      789 |      100 |      348 |       57 |     83.91% |48-50, 53-54, 283-\>286, 363-\>366, 371, 374, 394, 539-\>542, 544, 550-\>exit, 575, 654-\>657, 658, 661, 674-678, 714-715, 731-\>761, 742, 773, 807, 1079-\>1128, 1082-\>1085, 1086-\>1122, 1102-1103, 1108, 1129-\>1133, 1131-1132, 1176-\>1186, 1179-\>1186, 1181-1182, 1200, 1203-\>exit, 1402-1403, 1489, 1512, 1522-1523, 1526, 1538-\>1547, 1542-1546, 1552-1560, 1567, 1640, 1649, 1653-1654, 1688, 1691, 1694-1697, 1774-\>1776, 1782-1783, 1820, 1822, 1854-\>1858, 1884, 1890-1903, 1905-1907, 1914-1922, 1924, 1939-1940, 1950, 1967-\>1969, 1970-\>1972, 1973-\>1975, 1987-1990, 1992-1996, 2004, 2006, 2019-\>2024 |
| src/async\_batch\_llm/observers/\_\_init\_\_.py              |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/observers/base.py                      |       25 |        1 |        0 |        0 |     96.00% |        63 |
| src/async\_batch\_llm/observers/metrics.py                   |      139 |       12 |       70 |       14 |     85.65% |55, 72-\>79, 82-\>88, 89-\>exit, 114-\>108, 117-\>exit, 128, 134-135, 141, 142-\>147, 152, 155-160, 163, 169-\>exit |
| src/async\_batch\_llm/parallel.py                            |      415 |       27 |      110 |       10 |     92.95% |103-104, 107-108, 175, 215, 323, 334, 338, 346, 350, 354, 358, 362, 374, 481, 505-508, 546-547, 589-\>592, 655, 660, 755, 822, 879-\>884, 908-\>913, 960 |
| src/async\_batch\_llm/parsing.py                             |       63 |        0 |       18 |        0 |    100.00% |           |
| src/async\_batch\_llm/provider\_output.py                    |      102 |        1 |       28 |        1 |     98.46% |       225 |
| src/async\_batch\_llm/serialization.py                       |      249 |       41 |      102 |       23 |     81.20% |113, 116, 130, 139-140, 170, 222-223, 227, 252, 258, 268, 270, 350-351, 366, 376-377, 386, 388, 402, 473, 480, 487, 492, 510, 527, 563, 594-597, 649-650, 661-662, 668, 671-672, 684, 695 |
| src/async\_batch\_llm/single.py                              |       32 |        3 |        4 |        1 |     88.89% | 52-53, 67 |
| src/async\_batch\_llm/sqlite\_artifacts.py                   |      692 |       87 |      206 |       40 |     84.52% |86-87, 124-\>exit, 132, 137, 212-213, 215, 223-224, 265-266, 291, 301, 333-334, 366-371, 376, 380, 410, 419-420, 477-480, 500, 532, 551-554, 557-558, 560-\>567, 563-564, 575, 604, 612, 626, 632, 638-639, 661, 663-\>exit, 677-\>679, 680, 686-\>688, 697, 705-711, 748-749, 760-\>759, 775-\>774, 789-790, 802, 812-814, 830-\>845, 859, 879, 896-899, 903-905, 998-1000, 1016, 1083, 1090, 1093, 1097, 1108, 1139, 1235, 1301-1302, 1348-1349, 1357-1358, 1361-1362, 1366 |
| src/async\_batch\_llm/strategies/\_\_init\_\_.py             |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/strategies/errors.py                   |      143 |       16 |       38 |        5 |     88.40% |62-73, 160-\>exit, 193, 212, 403, 411, 481-482 |
| src/async\_batch\_llm/strategies/rate\_limit.py              |       31 |        0 |        2 |        0 |    100.00% |           |
| src/async\_batch\_llm/streaming.py                           |      199 |        1 |       70 |        4 |     98.14% |65-\>exit, 197-\>199, 253-\>256, 330, 336-\>exit |
| src/async\_batch\_llm/token\_estimation.py                   |       35 |        0 |        4 |        0 |    100.00% |           |
| src/async\_batch\_llm/token\_extractor.py                    |      128 |        4 |       50 |        5 |     94.94% |85-\>95, 98-\>104, 100-\>104, 143-144, 151, 198-\>200, 214 |
| **TOTAL**                                                    | **7579** |  **566** | **2390** |  **330** | **90.35%** |           |


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