#!/usr/bin/bash

sqlite3 /mnt/data-large/wikipedia/embeddings/wikipedia_metadata.db "SELECT * FROM articles ORDER BY idx DESC LIMIT 10;"

echo

sqlite3 /mnt/data-large/wikipedia/embeddings/wikipedia_metadata.db "SELECT COUNT(*) FROM articles;"
