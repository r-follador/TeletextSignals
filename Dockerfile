FROM pgvector/pgvector:pg18-trixie

# Seed the cluster with the SciChat database, extension, and tables.
COPY initdb/ /docker-entrypoint-initdb.d/
