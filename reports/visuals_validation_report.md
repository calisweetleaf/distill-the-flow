# Visual Intelligence Validation Report

- Status: **FAIL**
- Generated At: `2026-02-26T06:50:21.537944+00:00`
- Root: `D:\distill_the_flow`

## Artifacts
- `payload`: `D:\distill_the_flow\reports\visuals\atlas_payload.json`
- `dashboard`: `D:\distill_the_flow\reports\visuals\strategic_command_atlas.html`
- `manifest`: `D:\distill_the_flow\reports\visuals\visual_manifest.json`

## Checks
- `PASS` exists::atlas_payload.json
- `PASS` exists::strategic_command_atlas.html
- `PASS` exists::visual_manifest.json
- `PASS` payload_key::meta
- `PASS` payload_key::network
- `PASS` payload_key::phase_space
- `PASS` payload_key::truncation_surface
- `PASS` payload_key::cost_matrix
- `PASS` payload_key::interventions
- `FAIL` network_nonempty

## Terminal Log
- `[2026-02-26T06:50:21.476616+00:00] Starting visual intelligence validation run`
- `[2026-02-26T06:50:21.476616+00:00] Root directory: D:\distill_the_flow`
- `[2026-02-26T06:50:21.518376+00:00] Visual artifact generation succeeded`
- `[2026-02-26T06:50:21.518376+00:00] Check exists atlas_payload.json: True`
- `[2026-02-26T06:50:21.518376+00:00] Check exists strategic_command_atlas.html: True`
- `[2026-02-26T06:50:21.518376+00:00] Check exists visual_manifest.json: True`
- `[2026-02-26T06:50:21.537944+00:00] Check payload key 'meta': True`
- `[2026-02-26T06:50:21.537944+00:00] Check payload key 'network': True`
- `[2026-02-26T06:50:21.537944+00:00] Check payload key 'phase_space': True`
- `[2026-02-26T06:50:21.537944+00:00] Check payload key 'truncation_surface': True`
- `[2026-02-26T06:50:21.537944+00:00] Check payload key 'cost_matrix': True`
- `[2026-02-26T06:50:21.537944+00:00] Check payload key 'interventions': True`
- `[2026-02-26T06:50:21.537944+00:00] Check network density nodes=1, links=0`
- `[2026-02-26T06:50:21.537944+00:00] Validation failed: Network graph is empty`
