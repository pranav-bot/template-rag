[1] Parse document
     ↓
[2] Split into chunks (for LLM token limit)
     ↓
[3] For each chunk:
     ├─→ [3a] Extract section title and content using LLM
     ├─→ [3b] Search for similar section title in Vector Store
     │     ├─→ If similar title exists:
     │     │     ├─→ Check if similar content exists
     │     │     │     ├─→ If same → append new content
     │     │     │     └─→ If different → append
     │     │     └─→ else create new section
     ↓
[4] Save final section(s) into Vector Store


LET THE LLM figure out the sections

app1:
1.)extract sections and their chunk size
2,)then split according to chunk size

app2:
1.)extract sections and their chunk size
2,)then split according to whole document for that section

