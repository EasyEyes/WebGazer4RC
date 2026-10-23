# Security update: 2026-10-23

Pin Node.js 22.23.4 and Bun 1.4.4. Node engines also allow the patched 24.22.1 and 26.12.1 release lines. Update the committed binary Bun lockfile and compatible dependencies, including transitive security patches. The parent project's ignored, untracked npm lockfile is not authoritative and was neither staged nor deleted. CI and setup use Bun; no lockfile approval is needed.

[Node.js security advisory](https://nodejs.org/en/blog/vulnerability/october-2026-security-releases). [Bun dependency update behavior](https://bun.com/docs/pm/cli/update).

## Verification and limitations

Bun frozen installation passes with scripts disabled, and native audit reports no advisories. The parent remote-calibrator integration passes all 143 tests and its production build. No standalone test script is defined.

Native Bun audit: **0 critical** findings. 0 package names still have advisories. Remaining findings are deferred for compatibility/reachability review by 2026-10-30 and are not claimed harmless.


## Regression scenario

Use the pinned Node and Bun versions; run `bun install --frozen-lockfile --ignore-scripts` in WebGazer4RC and the parent. Review any blocked scripts before selectively enabling them. Run parent `bun run test` and `bun run build`. In a preview, complete camera permission, face/hand detection and distance calibration, exercise a new-object restart, and confirm gaze output remains finite. The hardware/browser scenario is planned, not executed. Re-run `bun audit` before release.
