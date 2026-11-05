---
source_file: "xlsx/basic.xlsx"
source_type: "xlsx"
extractor_version: "xlsx/0.1.0"
status: "warning"
---

# basic.xlsx

## Document metadata

- Author: Fixture Author
- Created: 2020-01-01T12:00:00

## Content

### Sheet: Data

| Item | Qty | Price | InStock | Added | Notes |
| --- | --- | --- | --- | --- | --- |
| Laptop | 2 | 1999.99 | TRUE | 2020-05-17 | with \| pipe |
| Mouse | 10 | 25.5 | FALSE | 2020-06-01 |  |
| Desk | 1 | 450 | TRUE | 2020-07-04 | unicode æøå |
|  |  |  |  |  |  |
| Total | =SUM(B2:B4) |  |  |  |  |
|  |  |  |  |  |  |
| Merged note |  |  |  |  |  |

### Sheet: Notes

|  |  |  |
| --- | --- | --- |
|  |  |  |
|  |  | Sparse sheet content |

## Extraction warnings

- **FORMULA_NO_CACHED_VALUE** (sheet Data): 1 formula cell(s) have no cached value; showing formula text only
