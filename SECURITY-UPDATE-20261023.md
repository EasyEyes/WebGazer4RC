# Security update: 2026-10-23

Node.js 22.23.4 is selected in `.nvmrc` and `engines.node`. This includes the October 21 security patches. Application source behavior is unchanged except for required build integration configuration.

Source: [Node.js October security releases](https://nodejs.org/en/blog/vulnerability/october-2026-security-releases).

## Verification and limitations

Node engine/local version updated only. Bun dependency audit reports unresolved findings. Parent remote-calibrator package-manager decision is pending; no lockfile migration was performed.

Dependency audit/remediation incomplete; see verification and blockers below.

## Verification scenario

Use the pinned Node version. Install from the authoritative lockfile with dependency scripts disabled first, review native/postinstall scripts, and run only those required by the package. Run the documented test/build commands. Repeat the native audit before release. In a preview, exercise the package's normal entry point and check browser/server errors. A successful build does not prove production integration.
