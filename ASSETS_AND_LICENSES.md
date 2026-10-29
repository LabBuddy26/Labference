# Assets and licences

Labference separates software licensing from asset licensing.

- First-party software is released under the MIT License.
- Published Labference dataset releases use CC BY 4.0 as described in
  `DATASET_LICENSE.md`.
- Every public asset requires an explicit record in
  `eai4s_eval/configs/assets/asset_licenses_v1.json`.
- `verified` means the source, copyright holder, licence and redistribution
  permission have been reviewed.
- `pending` / `NOASSERTION` means no public redistribution permission is
  claimed. The source file, rendered thumbnail and derivative demo media are
  excluded from public release artifacts.

The public website and public repository do not distribute chemistry
laboratory asset files or derivative thumbnails. Their appearance in an
internal development page does not grant permission to copy, download or
redistribute the underlying asset or a derivative rendering.

To approve an asset for a future release, record its copyright holder, source
URL, SPDX identifier, licence URL or bundled licence text, attribution,
modifications, and separate permissions for the source asset and derivative
media. The public release validator must pass before publication.
