# Publication steps

This folder is the intended GitHub repository root. Do not upload the enclosing internal project, virtual environments, original manuscript working files, authentication scripts, or the multi-gigabyte data archives.

1. Keep the owner's 2026-10-14 decision to defer software licensing: prepare the package without adding a software license grant. Add a license only if the owner later chooses to grant one. Preserve third-party notices and the separate data terms.
2. Fill the actual author list and repository URL in `CITATION.cff.template`, then rename it `CITATION.cff`. Do not advertise the reserved data DOI as a published paper or software DOI.
3. Publish the data record when its files and metadata are final, then verify public downloads without authentication. Until then, keep the README's explicit access-status note.
4. Reconcile the local Zenodo upload materials with the 2026-10-14 submission package: use the prepared `Zenodo_update_20261014` overlay in the enclosing project. It contains the current Methods S1, Data S2 and Data S3 plus updated README/checksums. The 27 large archives are unchanged. No remote record has been modified.
5. Keep `docs/VALIDATION.md` and `docs/FIGURES.md` honest about remaining gaps. Complete final R input assembly/rendering if full figure reproduction is a release requirement.
6. Create the GitHub repository and upload this folder's contents, including `.github`, `.gitignore` and the original source tree. The local ZIP is a convenience copy; GitHub should expose the unzipped source files.
7. Let the GitHub Actions workflow run. Only after it succeeds should its result be described as a passing GitHub CI run.

The 2026-10-14 release preparation does not push to GitHub or publish Zenodo. The original 2026-10-11 submission folder and the 2026-10-14 manuscript files are outside this export.
