import sqlite3

conn = sqlite3.connect('insa_sites.db')
cursor = conn.cursor()

# Export toute la table
with open('insa_sites.sql', 'w', encoding='utf-8') as f:
    for line in conn.iterdump():
        f.write('%s\n' % line)

conn.close()