# Bruno smoke tests

A read-only HTTP smoke suite for a **running** DeepFellow Infra instance. It
checks that a deployment is actually up, serving its docs, reporting the version
it is supposed to, and rejecting unauthorized calls.

Nothing here mutates state, so it is safe to point at any environment —
including shared ones — and to re-run as often as you like.

This complements `tests/unit/`, which exercises the ASGI app in-process. Only a
real HTTP run catches breakage in the proxy, TLS, env wiring, or the container
failing to serve at all.

## Layout

| Folder | Covers |
|--------|--------|
| `smoke/` | `GET /health` (no auth), `GET /docs`, `GET /openapi.json`, `GET /info` (server key + version) |
| `auth/` | Negatives: missing token, invalid token, and the server key being refused on `/admin/*` and `/metrics` |
| `api/` | `GET /v1/models` with the server key |
| `admin/` | Admin-key reads: settings, services, all models, config entries, CPU/RAM and GPU stats, mesh info and topology |

Assertions check response shape, never environment-specific content — how many
services or models are installed differs per environment, so those lists are
only checked for being arrays. `admin/gpu-stats` asserts the status alone: the
endpoint returns `GpuStats | None` and is `null` on a host without a GPU.

`environments/` holds the target hosts for the Bruno GUI (`local`, `test`); CI
injects the host with `--env-var host=...` instead.

## Running locally

The helper starts a dev server if one isn't already up, runs the suite, and
tears it back down:

```bash
npm install -g @usebruno/cli@4.0.0   # once — same version CI pins
tests/smoke/run-local.sh             # whole suite
tests/smoke/run-local.sh smoke       # just the smoke/ folder
```

The version is pinned so a new Bruno release can't change CLI behaviour under
the pipeline. To move it, bump `BRUNO_CLI_VERSION` in `.gitlab-ci.yml` and this
line together, and re-run the suite locally against the new version first.

It resolves the auth tokens the way the server itself does, which is worth
knowing if you ever see unexplained 401s:

- `ADMIN_TOKEN` ← `DF_INFRA_ADMIN_API_KEY` in `.env` (a bootstrap setting).
- `API_TOKEN` ← `storage/config.json` if it exists, otherwise `DF_INFRA_API_KEY`
  in `.env`. The server key is dynamic config: once `config.json` exists the
  `.env` value is no longer read (see `server/dynamic_config.py`), so a stale
  `.env` will not match the running server.

`EXPECTED_VERSION` is set from `pyproject.toml` so the `/info` version assertion
is meaningful locally too.

### Against another environment

```bash
export API_TOKEN=... ADMIN_TOKEN=...
bru run tests/smoke --env-var host=https://dfinfra.test.simplito.com
```

Set `EXPECTED_VERSION` as well to assert a specific released version; leave it
unset and `/info` is only checked for shape.

## In CI

The `smoke-test` job (`.gitlab-ci.yml`) sits in its own stage between `deploy`
and `release`, and runs after every `deploy_to_dev` — on `main`, `hotfix-*` and
`v*.*.*` tags — so a regression shows up on the merge that caused it.

On a tag it doubles as the **blocking release gate**: a failure stops
`push_to_github`, so a broken build is never published. There it polls
`/health`, then waits for `/info` to report the tagged version before asserting
— otherwise it could race the rollout and green-light the *previous* image. On
`main` there is no expected version to wait for, so the run may exercise the
build that was already deployed; every other check still holds.

It needs `DF_INFRA_API_KEY` and `DF_INFRA_ADMIN_API_KEY` as **masked** CI/CD
variables matching the target instance, and `DF_SMOKE_URL` for the host.
Results are exported as JUnit.

## Adding a request

Create a `.bru` file in the folder that fits, give it the next `seq`, and assert
on the **body**, not just the status. `server/api/fallback.py` serves
`index.html` with status 200 for any unmatched path, so a bare
`expect(res.getStatus()).to.equal(200)` also passes when the route has vanished
entirely.
