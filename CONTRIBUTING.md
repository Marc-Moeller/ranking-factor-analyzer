# Contributing

Bug reports, factor proposals and pull requests are all welcome.

## Sign your commits (DCO)

Every commit must carry a `Signed-off-by` line. Git adds it for you:

```bash
git commit -s -m "add hreflang consistency factor"
```

That line certifies you wrote the patch or otherwise have the right to submit it, under the
[Developer Certificate of Origin 1.1](https://developercertificate.org/). We cannot merge unsigned
commits — see [`LICENSING.md`](LICENSING.md) for why the sign-off matters here specifically.

## Before you open a PR

```bash
pip install -r requirements.txt
pytest                     # the smoke suite must pass
ruff check ranklens api    # if you have ruff; we are not strict about style
```

Run the analyzer against a real keyword and read the report. Most bugs in this codebase are
"the number is wrong", not "it crashed", and only a real run surfaces those.

## Adding a ranking factor

Factors live in `ranklens/factors_registry.py` as `FactorDef` entries. A good factor:

- **Is measurable from a fetched page, a SERP result, or a corpus of both.** If measuring it needs a
  paid API, gate it behind a flag the way `include_backlinks` works.
- **Has a defensible direction.** Say whether more is better, and why. "Correlates with position" is
  not a reason on its own; half the things that correlate with rank are downstream of it.
- **Degrades to `None`, never to zero.** A missing measurement and a measured zero mean different
  things, and the correlation maths treats them differently.

Add the factor, run the analyzer on a handful of keywords, and put the before/after in the PR. A
factor that never fires on real SERPs is worse than no factor.

## What we will probably decline

- Vendor-specific integrations that only work with one paid provider, unless they sit behind the
  same optional-credential pattern everything else uses.
- Changes that make a run require more paid API calls by default. Defaults must stay runnable by
  someone with a single LLM key.
- Scrapers for platforms whose terms clearly forbid it.
