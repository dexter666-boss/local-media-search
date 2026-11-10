# Contributing

Before changing search behavior, record a reproducible query, expected result, and actual result. Add a regression test when appropriate and run `python -m unittest discover -v`.

For bug reports, include your OS, Python version, CPU/GPU, query, search mode, and sanitized error text. Do not attach private media, credentials, cookies, or full personal folder paths. Share only media you have permission to distribute.

Model accuracy improvements need both false-positive and false-negative checks on held-out material. Do not add filename or private-dataset rules to make a benchmark pass. Keep score meanings visible and avoid unsupported accuracy claims.

Contributions are welcome under the repository MIT license. Keep changes focused and describe validation and known limitations.
