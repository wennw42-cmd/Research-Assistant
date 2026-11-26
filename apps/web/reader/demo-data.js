window.__READER_DOC__ = {
  "version": 1,
  "meta": {
    "title": "A Demo Paper: Attention-Based Sequence Models",
    "source": {
      "path": "demo.pdf",
      "sha1": "demo",
      "pages": 2,
      "parser": "demo"
    },
    "layout": "double",
    "pages": 2
  },
  "pages": [
    {
      "page": 1,
      "width": 595.0,
      "height": 842.0,
      "columns": 2,
      "gutters": [
        [
          290.0,
          305.0
        ]
      ],
      "estimated": false
    },
    {
      "page": 2,
      "width": 595.0,
      "height": 842.0,
      "columns": 2,
      "gutters": [
        [
          290.0,
          305.0
        ]
      ],
      "estimated": false
    }
  ],
  "blocks": [
    {
      "id": "p1b1",
      "anchor": "blk-p1b1",
      "type": "title",
      "page": 1,
      "bbox": [
        120.0,
        48.0,
        475.0,
        82.0
      ],
      "column": -1,
      "span": true,
      "rank": 0,
      "text": "A Demo Paper: Attention-Based Sequence Models",
      "latex_approx": false,
      "confidence": 0.99,
      "segments": [
        {
          "kind": "text",
          "text": "A Demo Paper: Attention-Based Sequence Models"
        }
      ]
    },
    {
      "id": "p1b2",
      "anchor": "blk-p1b2",
      "type": "paragraph",
      "page": 1,
      "bbox": [
        60.0,
        112.0,
        280.0,
        190.0
      ],
      "column": 0,
      "span": false,
      "rank": 1,
      "text": "We propose a compact attention model [1]. Prior sequence models rely on recurrence, which limits parallel training.",
      "latex_approx": false,
      "confidence": 0.95,
      "segments": [
        {
          "kind": "text",
          "text": "We propose a compact attention model "
        },
        {
          "kind": "cite",
          "text": "[1]",
          "occurrence_id": "c-p1b2-37",
          "ref_ids": [
            "1"
          ],
          "targets": [
            "ref-p2b4"
          ],
          "status": "resolved"
        },
        {
          "kind": "text",
          "text": ". Prior sequence models rely on recurrence, which limits parallel training."
        }
      ]
    },
    {
      "id": "p1b3",
      "anchor": "blk-p1b3",
      "type": "heading",
      "page": 1,
      "bbox": [
        60.0,
        210.0,
        280.0,
        226.0
      ],
      "column": 0,
      "span": false,
      "rank": 2,
      "level": 1,
      "number": "1",
      "text": "1 Introduction",
      "latex_approx": false,
      "confidence": 0.95,
      "segments": [
        {
          "kind": "text",
          "text": "1 Introduction"
        }
      ]
    },
    {
      "id": "p1b4",
      "anchor": "blk-p1b4",
      "type": "paragraph",
      "page": 1,
      "bbox": [
        60.0,
        236.0,
        280.0,
        326.0
      ],
      "column": 0,
      "span": false,
      "rank": 3,
      "text": "Attention scales with sequence length rather than depth [2–3]. This property is central to our design.",
      "latex_approx": false,
      "confidence": 0.95,
      "segments": [
        {
          "kind": "text",
          "text": "Attention scales with sequence length rather than depth "
        },
        {
          "kind": "cite",
          "text": "[2–3]",
          "occurrence_id": "c-p1b4-56",
          "ref_ids": [
            "2",
            "3"
          ],
          "targets": [
            "ref-p2b5",
            "ref-p2b6"
          ],
          "status": "resolved"
        },
        {
          "kind": "text",
          "text": ". This property is central to our design."
        }
      ]
    },
    {
      "id": "p1b5",
      "anchor": "blk-p1b5",
      "type": "paragraph",
      "page": 1,
      "bbox": [
        315.0,
        112.0,
        535.0,
        190.0
      ],
      "column": 1,
      "span": false,
      "rank": 4,
      "text": "The method builds on prior encoder-decoder work [2, 3].",
      "latex_approx": false,
      "confidence": 0.95,
      "segments": [
        {
          "kind": "text",
          "text": "The method builds on prior encoder-decoder work "
        },
        {
          "kind": "cite",
          "text": "[2, 3]",
          "occurrence_id": "c-p1b5-48",
          "ref_ids": [
            "2",
            "3"
          ],
          "targets": [
            "ref-p2b5",
            "ref-p2b6"
          ],
          "status": "resolved"
        },
        {
          "kind": "text",
          "text": "."
        }
      ]
    },
    {
      "id": "p1b6",
      "anchor": "blk-p1b6",
      "type": "heading",
      "page": 1,
      "bbox": [
        315.0,
        210.0,
        535.0,
        226.0
      ],
      "column": 1,
      "span": false,
      "rank": 5,
      "level": 1,
      "number": "2",
      "text": "2 Method",
      "latex_approx": false,
      "confidence": 0.95,
      "segments": [
        {
          "kind": "text",
          "text": "2 Method"
        }
      ]
    },
    {
      "id": "p1b7",
      "anchor": "blk-p1b7",
      "type": "paragraph",
      "page": 1,
      "bbox": [
        315.0,
        236.0,
        535.0,
        300.0
      ],
      "column": 1,
      "span": false,
      "rank": 6,
      "text": "Each layer applies scaled dot-product attention. We compare against the large-batch baseline [9].",
      "latex_approx": false,
      "confidence": 0.95,
      "segments": [
        {
          "kind": "text",
          "text": "Each layer applies scaled dot-product attention. We compare against the large-batch baseline "
        },
        {
          "kind": "cite",
          "text": "[9]",
          "occurrence_id": "c-p1b7-93",
          "ref_ids": [
            "9"
          ],
          "targets": [],
          "status": "missing"
        },
        {
          "kind": "text",
          "text": "."
        }
      ]
    },
    {
      "id": "p1b8",
      "anchor": "blk-p1b8",
      "type": "heading",
      "page": 1,
      "bbox": [
        315.0,
        316.0,
        535.0,
        332.0
      ],
      "column": 1,
      "span": false,
      "rank": 7,
      "level": 2,
      "number": "2.1",
      "text": "2.1 Scaled Attention",
      "latex_approx": false,
      "confidence": 0.95,
      "segments": [
        {
          "kind": "text",
          "text": "2.1 Scaled Attention"
        }
      ]
    },
    {
      "id": "p1b9",
      "anchor": "blk-p1b9",
      "type": "paragraph",
      "page": 1,
      "bbox": [
        315.0,
        342.0,
        535.0,
        410.0
      ],
      "column": 1,
      "span": false,
      "rank": 8,
      "text": "The softmax temperature follows the original formulation [1].",
      "latex_approx": false,
      "confidence": 0.95,
      "segments": [
        {
          "kind": "text",
          "text": "The softmax temperature follows the original formulation "
        },
        {
          "kind": "cite",
          "text": "[1]",
          "occurrence_id": "c-p1b9-57",
          "ref_ids": [
            "1"
          ],
          "targets": [
            "ref-p2b4"
          ],
          "status": "resolved"
        },
        {
          "kind": "text",
          "text": "."
        }
      ]
    },
    {
      "id": "p1b10",
      "anchor": "blk-p1b10",
      "type": "table",
      "page": 1,
      "bbox": [
        60.0,
        450.0,
        535.0,
        540.0
      ],
      "column": -1,
      "span": true,
      "rank": 9,
      "text": "Table 1: Results on the demo benchmark [3].",
      "latex_approx": false,
      "markdown": "| Model | BLEU |\n| --- | --- |\n| Ours | 28.4 |\n| Baseline | 25.2 |",
      "caption": "Table 1: Results on the demo benchmark [3].",
      "confidence": 0.92,
      "source": "find_tables",
      "segments": [
        {
          "kind": "text",
          "text": "Table 1: Results on the demo benchmark "
        },
        {
          "kind": "cite",
          "text": "[3]",
          "occurrence_id": "c-p1b10-39",
          "ref_ids": [
            "3"
          ],
          "targets": [
            "ref-p2b6"
          ],
          "status": "resolved"
        },
        {
          "kind": "text",
          "text": "."
        }
      ]
    },
    {
      "id": "p2b1",
      "anchor": "blk-p2b1",
      "type": "heading",
      "page": 2,
      "bbox": [
        60.0,
        60.0,
        280.0,
        76.0
      ],
      "column": 0,
      "span": false,
      "rank": 10,
      "level": 1,
      "number": "3",
      "text": "3 Results",
      "latex_approx": false,
      "confidence": 0.95,
      "segments": [
        {
          "kind": "text",
          "text": "3 Results"
        }
      ]
    },
    {
      "id": "p2b2",
      "anchor": "blk-p2b2",
      "type": "paragraph",
      "page": 2,
      "bbox": [
        60.0,
        86.0,
        280.0,
        160.0
      ],
      "column": 0,
      "span": false,
      "rank": 11,
      "text": "The proposed model improves BLEU over the baseline [2].",
      "latex_approx": false,
      "confidence": 0.95,
      "segments": [
        {
          "kind": "text",
          "text": "The proposed model improves BLEU over the baseline "
        },
        {
          "kind": "cite",
          "text": "[2]",
          "occurrence_id": "c-p2b2-51",
          "ref_ids": [
            "2"
          ],
          "targets": [
            "ref-p2b5"
          ],
          "status": "resolved"
        },
        {
          "kind": "text",
          "text": "."
        }
      ]
    }
  ],
  "toc": [
    {
      "id": "toc-2",
      "text": "1 Introduction",
      "level": 1,
      "number": "1",
      "page": 1,
      "anchor": "blk-p1b3",
      "kind": "heading",
      "children": []
    },
    {
      "id": "toc-5",
      "text": "2 Method",
      "level": 1,
      "number": "2",
      "page": 1,
      "anchor": "blk-p1b6",
      "kind": "heading",
      "children": [
        {
          "id": "toc-7",
          "text": "2.1 Scaled Attention",
          "level": 2,
          "number": "2.1",
          "page": 1,
          "anchor": "blk-p1b8",
          "kind": "heading",
          "children": []
        }
      ]
    },
    {
      "id": "toc-11",
      "text": "3 Results",
      "level": 1,
      "number": "3",
      "page": 2,
      "anchor": "blk-p2b1",
      "kind": "heading",
      "children": []
    },
    {
      "id": "toc-13",
      "text": "References",
      "level": 1,
      "number": null,
      "page": 2,
      "anchor": "reader-references",
      "kind": "references",
      "children": []
    }
  ],
  "references": [
    {
      "id": "ref-p2b4",
      "block_id": "p2b4",
      "number": "1",
      "anchor": "ref-p2b4",
      "page": 2,
      "bbox": [
        60.0,
        220.0,
        280.0,
        258.0
      ],
      "text": "[1] A. Author. A Demo Paper on Attention. arXiv:1706.03762, 2017.",
      "doi": null,
      "arxiv": "arXiv:1706.03762",
      "url": null,
      "ambiguous": false,
      "cite_ids": [
        "c-p1b2-37",
        "c-p1b9-57"
      ]
    },
    {
      "id": "ref-p2b5",
      "block_id": "p2b5",
      "number": "2",
      "anchor": "ref-p2b5",
      "page": 2,
      "bbox": [
        315.0,
        220.0,
        535.0,
        258.0
      ],
      "text": "[2] B. Author. Sequence Models Revisited. DOI:10.1000/demo.2018, 2018.",
      "doi": "10.1000/demo.2018",
      "arxiv": null,
      "url": null,
      "ambiguous": false,
      "cite_ids": [
        "c-p1b4-56",
        "c-p1b5-48",
        "c-p2b2-51"
      ]
    },
    {
      "id": "ref-p2b6",
      "block_id": "p2b6",
      "number": "3",
      "anchor": "ref-p2b6",
      "page": 2,
      "bbox": [
        60.0,
        264.0,
        280.0,
        302.0
      ],
      "text": "[3] C. Author. Attention Baselines. https://example.org/p3, 2019.",
      "doi": null,
      "arxiv": null,
      "url": "https://example.org/p3",
      "ambiguous": false,
      "cite_ids": [
        "c-p1b4-56",
        "c-p1b5-48",
        "c-p1b10-39"
      ]
    }
  ],
  "citations": {
    "occurrences": [
      {
        "id": "c-p1b2-37",
        "block_id": "p1b2",
        "start": 37,
        "end": 40,
        "raw": "[1]",
        "ref_ids": [
          "1"
        ],
        "targets": [
          "ref-p2b4"
        ],
        "unresolved_ids": [],
        "ambiguous_ids": [],
        "status": "resolved"
      },
      {
        "id": "c-p1b4-56",
        "block_id": "p1b4",
        "start": 56,
        "end": 61,
        "raw": "[2–3]",
        "ref_ids": [
          "2",
          "3"
        ],
        "targets": [
          "ref-p2b5",
          "ref-p2b6"
        ],
        "unresolved_ids": [],
        "ambiguous_ids": [],
        "status": "resolved"
      },
      {
        "id": "c-p1b5-48",
        "block_id": "p1b5",
        "start": 48,
        "end": 54,
        "raw": "[2, 3]",
        "ref_ids": [
          "2",
          "3"
        ],
        "targets": [
          "ref-p2b5",
          "ref-p2b6"
        ],
        "unresolved_ids": [],
        "ambiguous_ids": [],
        "status": "resolved"
      },
      {
        "id": "c-p1b7-93",
        "block_id": "p1b7",
        "start": 93,
        "end": 96,
        "raw": "[9]",
        "ref_ids": [
          "9"
        ],
        "targets": [],
        "unresolved_ids": [
          "9"
        ],
        "ambiguous_ids": [],
        "status": "missing"
      },
      {
        "id": "c-p1b9-57",
        "block_id": "p1b9",
        "start": 57,
        "end": 60,
        "raw": "[1]",
        "ref_ids": [
          "1"
        ],
        "targets": [
          "ref-p2b4"
        ],
        "unresolved_ids": [],
        "ambiguous_ids": [],
        "status": "resolved"
      },
      {
        "id": "c-p1b10-39",
        "block_id": "p1b10",
        "start": 39,
        "end": 42,
        "raw": "[3]",
        "ref_ids": [
          "3"
        ],
        "targets": [
          "ref-p2b6"
        ],
        "unresolved_ids": [],
        "ambiguous_ids": [],
        "status": "resolved"
      },
      {
        "id": "c-p2b2-51",
        "block_id": "p2b2",
        "start": 51,
        "end": 54,
        "raw": "[2]",
        "ref_ids": [
          "2"
        ],
        "targets": [
          "ref-p2b5"
        ],
        "unresolved_ids": [],
        "ambiguous_ids": [],
        "status": "resolved"
      }
    ],
    "by_number": {
      "1": [
        "ref-p2b4"
      ],
      "2": [
        "ref-p2b5"
      ],
      "3": [
        "ref-p2b6"
      ]
    },
    "unresolved_numbers": [
      "9"
    ],
    "stats": {
      "total": 7,
      "resolved": 6,
      "partial": 0,
      "missing": 1,
      "ambiguous": 0,
      "invalid": 0
    }
  },
  "stats": {
    "blocks": 12,
    "references": 3,
    "toc_nodes": 5,
    "pages": 2,
    "citations": 7
  },
  "warnings": [
    "正文引用了参考文献表中不存在的编号：9"
  ]
};
