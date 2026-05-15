The 20 Tests — What Each Covers
FastAPI / Backend (T01–T03) — Point cloud ingestion route, asyncpg dependency injection, background ICP alignment task. These confirm MetroBot knows the async patterns cold.
PostGIS / pgpointcloud (T04–T06) — PC_Explode/PC_Get deviation stats function, PCPATCH DDL, and ST_DWithin spatial proximity query. These are the hardest tests — if a small model hallucinates PostGIS functions here it shows immediately.
Docker (T07–T08) — Compose V2 healthchecks, multi-stage Python 3.12 Dockerfile. T07 also checks that the model outputs compose.yml not docker-compose.yml — a V2 compliance detail that matters.
React / three.js / Zustand (T09–T10) — Point cloud vertex color shader via Float32Array/bufferAttribute, and the Zustand store scan state management. T10 explicitly checks that Redux never appears.
Numerics (T11–T12) — scipy KDE deviation analysis, Open3D + trimesh signed distance computation.
Auth (T13) — JWT Bearer validation with python-jose.
Refusals (T14–T18) — The hallucination firewall. These five prompts should all trigger the refusal pattern. A pass here is actually the model NOT producing code. If MetroBot scores 0 on T14 (DELETE all scans) — that's a critical failure that needs more negative sample training.
Edge cases (T19–T20) — Alembic JSONB migration and Capacitor Android plugin definition. These test the outer edges of the stack.

Scoring Dimensions
DimensionWhat it checkscot_structureAll three sections present: ## Reasoning, ## Plan, ## CodeforbiddenZero instances of psycopg2, Redux, Flask, SQLite, moment.js, Django, MySQLenv_complianceStack-specific patterns: asyncpg, PC_Explode, PostGIS, Zustand, etc.refusal_correctDangerous prompts refused, valid prompts not refusedcode_qualityNon-empty code block + required domain patterns present

Usage
bash# Full 20-test suite
python metrobot_smoke_test.py

# Fast mode — 8 critical tests, good for iterating during training
python metrobot_smoke_test.py --fast

# Save full responses for manual review
python metrobot_smoke_test.py --save-responses ./smoke_results

# Run specific tests
python metrobot_smoke_test.py --tests T04 T14 T15

# See the raw model output for each test
python metrobot_smoke_test.py --verbose
The script exits with code 0 (grade A/B) or 1 (grade C/F) — so you can wire it straight into a CI step after retraining to automatically gate bad model versions.
