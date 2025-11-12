#!/usr/bin/bash

sqlite3 /mnt/data/wikipedia/embeddings/metadata.db "SELECT * FROM articles ORDER BY idx DESC LIMIT 10;"

echo

sqlite3 /mnt/data/wikipedia/embeddings/metadata.db "SELECT COUNT(*) FROM articles;"
