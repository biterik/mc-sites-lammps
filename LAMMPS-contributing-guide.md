# Getting your own developments into LAMMPS — a working guide

> Scope: reference note for any cowork/LLM session (or human) preparing to
> upstream code into LAMMPS. Distills the **official process** *and* the
> **unwritten expectations of the maintainers — above all Axel Kohlmeyer
> (`akohlmey`)**, who reviews the largest share of contributions and effectively
> sets the house style. Read this before writing a pair/fix/compute you intend
> to submit; following it up front turns a months-long back-and-forth into a
> merge that can happen in hours.
>
> Last updated: 2026-07-07. Sources listed at the bottom. Items the LAMMPS docs
> mark **(strict)** are non-negotiable; **(preferred)/(varied)** are negotiable
> but you should have a reason to deviate.

---

## 0. TL;DR — what actually gets your PR merged fast

1. **Talk to the developers *before* writing anything non-trivial** (GitHub issue
   or Slack). Coordinating up front is the single biggest time-saver and avoids
   being scooped or duplicating work.
2. **Make it an add-on, not a core change.** Design so your feature is new files
   dropped into a package, requiring at most trivial core edits (e.g. making a
   method `virtual`). Substantive core changes trigger long review.
3. **Put it in the right package** (or a new `FOO` package with a README). New
   styles are almost never accepted directly into `src/`.
4. **It must compile and run correctly** with both `-DLAMMPS_SMALLBIG` (default)
   *and* `-DLAMMPS_BIGBIG`, in **serial and parallel (MPI)**, against the current
   `develop` branch — and **pass the GitHub CI**. This is the hard gate.
5. **Ship documentation** (`.rst` in `doc/src/`, American English, ASCII-only)
   and a **small, fast example**. No docs = no merge.
6. **Match the house style** (naming, includes, `clang-format`, constants, etc.
   — see §5). This is what reviewers, especially Axel, nitpick.
7. **License is GPL-2.0-only** with the LAMMPS copyright header on every new file.

Everything below expands these.

---

## 1. The mental model (what the maintainers actually care about)

LAMMPS is a large package where **most code is contributed by non-professional
programmers** (domain scientists). The maintainers' stated goal is software that
is "versatile, reliable, high-quality, efficient, portable, and easy to maintain
and modify." Every rule below exists to protect **maintainability and the
parallel performance of the core** against a flood of well-meaning but
inconsistent contributions.

Practical consequences of that worldview, worth internalizing:

- **The maintainers are volunteers and will not do your work for you.** As
  Kohlmeyer has put it on the forum: *"this is the 'cruel' world of open source
  software. If you want it done, you either have to do it yourself … convince
  somebody to do it for you (make a more compelling argument), or pay somebody to
  do it."* A feature request without an implementer may just get labeled
  `volunteer_needed` and sit.
- **The burden of quality is on the contributor.** Reviewers will push back on
  style, docs, and portability even when the science is fine. Don't take it
  personally; consistency across thousands of files is the whole point.
- **"It works on my machine" is not enough.** BIGBIG + MPI + current `develop` +
  CI is the real bar, precisely because contributors tend to test only their one
  configuration.
- **Smaller, modular, self-contained contributions get merged faster.** A big
  monolithic drop that touches core files is the slow path.

---

## 2. Before you write code — coordinate

- For anything beyond a trivial single-file style, **open a GitHub issue** (or
  comment on an existing one) describing what you plan to implement. This lets
  developers flag conflicts, suggest a cleaner integration, or tell you it belongs
  as a separate tool rather than in LAMMPS.
- For informal design discussion you can request access to the **LAMMPS
  developer Slack** (`lammps.slack.com`) by emailing `slack@lammps.org` —
  *development topics only*; it is **not** a user-support channel.
- **User questions go to the MatSci Discourse forum** (`matsci.org/lammps`),
  **never** the GitHub issue tracker. Filing "how do I use X" as an issue is a
  classic way to annoy the maintainers.
- Whether a suggested feature is accepted depends on: difficulty, maintenance
  effort, breadth of user benefit, whether a developer understands the underlying
  physics, and whether it *fits* a code like LAMMPS at all. **Argue the benefit
  convincingly** — a well-formulated proposal matters.

---

## 3. Where your code goes

- **Not `src/`.** New styles in the core `src` folder "are rarely accepted."
- **Find the right existing package** (see `Packages_details.html`) whose general
  purpose your feature fits. If it fits nothing well, it may go in an `EXTRA-*`
  package or `MISC`.
- **Many related features, or a dependency on a library** (bundled or external) →
  **create a new package** with its own directory (name like `FOO`, uppercase).
  Include a `README` text file with your name, contact info, and a short
  description of what the package does.
- **External option:** you don't *have* to upstream. You can distribute an add-on
  yourself (website, your own repo). The LAMMPS team will advertise it on their
  "External LAMMPS packages and tools" page if you email them. Recommended naming
  for external packages: `USER-<name>` to distinguish from in-distribution
  packages (which no longer carry the old `USER-` prefix).

---

## 4. The strict gates (these block merging)

**Licensing (strict).** By submitting a PR you agree your contribution ships
under the **GNU GPL version 2 — only, not 3+** (an LGPL-2.1 variant exists on
request). Put the LAMMPS copyright + GPL notice, then your name and email, at the
top of every new source file. Incompatible-licensed code must be split into a
GPL-compatible core part in `src` plus a separately downloaded/compiled external
library (note: split licensing complicates binary packaging).

**Integration testing / portability (strict).** Code **must**:
- compile against the **current `develop`** version, containing its latest bugfixes;
- work correctly with **both** `-DLAMMPS_SMALLBIG` (default) and `-DLAMMPS_BIGBIG`;
- work in **serial and in parallel via MPI**;
- **pass the automated GitHub / `ci.lammps.org` CI** (compiles in many configs,
  runs unit tests, builds the docs to HTML *and* PDF). CI re-runs on every push
  to the PR branch. **Nothing merges with failing CI.**

**Documentation (strict).**
- New/changed styles need matching docs as **reStructuredText `.rst` in
  `doc/src/`**, in **American English**, **ASCII-only** (special chars via inline
  LaTeX math so the PDF build works).
- Register new commands in the sphinx command tables/lists; new packages need
  their list entries + a package description (+ build instructions if needed).
- `make html` **and** `make spelling` must run clean (no warnings); add genuine
  false positives to `doc/utils/sphinx-config/false_positives.txt`.
- Citation labels must be unique across **all** `.rst` files. Use a
  "Restrictions" note if the command needs a particular package.
- Public C++/Fortran API changes need **doxygen** comments + Programmer-Guide
  updates. Complex features benefit from a **Howto** page.
- Rule of thumb Axel repeats: *the clearer your docs, README, and examples, the
  more likely people actually use your feature.*

**Language standards (strict).**
- Core is **C++17**, written as "C with classes." **Avoid** operator overloading
  and heavy templates; the code must stay readable to scientists with limited C++.
  `std::string`, `auto`, and cautious `std::vector` are fine. Post-C++17 code is
  only accepted if confined to an optional package.
- Bundled Fortran must be **Fortran 2003** (and ideally rewritten as C++ — as of
  2023 the executable has no Fortran left). Python must target **3.6** unless a
  later version is documented.
- **Old-standard compatibility is deliberate** — HPC clusters run old toolchains
  and LAMMPS users must still build there. (Directly relevant to our own cluster
  work; don't assume a modern compiler.)

**Build system (strict).** Two systems: traditional **Makefiles** and **CMake**.
As of fall 2025 supporting GNU make is no longer required — new packages may be
**CMake-only**. A single independent header/impl pair usually just needs adding to
`src/.gitignore`. Dependencies on other packages need `Install.sh` +
`src/Depend.sh` edits (make) and/or `cmake/CMakeLists.txt`, `cmake/presets`, and a
`cmake/Modules/Packages/` file (CMake). Copy an existing package as a template.

**Naming (strict).** User-visible command/style names: **all lowercase**, only
letters/numbers/forward-slashes, descriptive, no obscure initialisms (except
established ones like `lj`). A compute `some/name` → files
`compute_some_name.{h,cpp}`, guard `LMP_COMPUTE_SOME_NAME_H`, class
`ComputeSomeName`. This mapping is mechanical and reviewers *will* enforce it.

---

## 5. House style — the stuff Axel nitpicks (Modify_style)

These are where reviews get pedantic. Getting them right pre-emptively is the
fastest way to a happy reviewer. Representative reference files:
`pair_lj_cut.{h,cpp}`, `utils.{h,cpp}`.

**Include files & headers (strict-ish).**
- A "style" header (one with a `SomeStyle(name,Class);` macro) must include
  **only the base-class header**, and otherwise use **forward declarations +
  pointers**. For libraries use **PIMPL**. This is a *strict* rule — it's where
  cross-package type clashes and nasty bugs came from historically.
- Headers **must not** contain `using` statements and should include the absolute
  minimum. Put includes in the `.cpp`, "include what you use."
- `pointers.h` already pulls in: `mpi.h, cstddef, cstdio, cstdlib, string,
  utils.h, vector, fmt/format.h, climits, cinttypes` — so `FILE/NULL/INT_MAX`
  etc. are available; don't re-include them.
- **Don't initialize member variables in the header** — use the constructor's
  initializer list / body. **Pointer members must be initialized to `nullptr`**
  in the initializer list. Comment members that carry meaning.
- Angle brackets for system/library headers, quotes for local; use C++ names
  (`<cstdlib>` not `<stdlib.h>`).
- `#include` order in `some_name.cpp`: own header + blank line → LAMMPS headers
  alphabetically + blank line → system headers + blank line → `using namespace`.

**Whitespace (preferred).** No TABs (except where syntax demands, e.g.
makefiles), no trailing whitespace, **Unix LF** line endings, newline at EOF.
The `tools/coding_standard` python scripts detect (and with `-f` auto-fix) these.

**Constants (strongly preferred).** Use `static constexpr` (UPPERCASE names),
**not** `#define`. Typed, warns if unused, no surprise text substitution (which
has genuinely broken package combinations before). Reuse `MathConst` /
`EwaldConst`. Preprocessor only for real macros/conditional compilation, and
never in headers.

**Braces / formatting (strongly preferred).** A `.clang-format` (clang-format 8+)
is provided; run `clang-format -i new-file.cpp` on **your** files (not others').
Protect hard-to-format blocks with `// clang-format off` / `on`. The
`SomeStyle(keyword,Class);` macro in a style header **must** be wrapped in
`// clang-format off/on` and end with a semicolon.

**Miscellaneous (varied but expected).**
- I/O via C **stdio**, not iostreams.
- **No "alternative tokens"** (`and`/`or`/`not`) — use `&&`/`||`/`!` (not all
  compilers enable them).
- Screen/logfile output only on **MPI rank 0**; prefer `utils::logmesg()`.
- `virtual`/`override`/`final` per C++ Core Guideline C.128 (`virtual` only for a
  new virtual, `override` when overriding, `final` to stop overriding).
- **Don't write empty/default destructors** (`~A() override {}` or `= default`) —
  let the compiler generate them.
- Files should be **0644** (0755 only for actual scripts).

**Error messages (preferred, newer policy since 4May2022).** Prefer **specific,
self-explanatory** error strings via the `{fmt}`-style `Error` methods (format
string + args), kept to ~1–2 lines. For multi-cause errors add a paragraph to the
`Error_details` page with an `_errNNNN:` anchor and reference it via
`utils::errorurl()`. Use `utils::missing_cmd_args()` for missing-argument errors.

---

## 6. Examples & tests

**Examples (preferred).** Include a **small, fast (1-CPU) example** under
`examples/` or `examples/PACKAGES/`. Rules: comment out output-generating commands
(unless output *is* the point, e.g. a new compute); do **not** use `log`, `echo`,
`package`, `processors`, `suffix` in the input (exception: `processors * * 1`);
name files `in.name`, `data.name`, `log.version.name.<compiler>.<ncpu>`; keep
total size small; **symlink** shared potential/data files rather than copy; no
CPU/time fields in custom `thermo_style`.

**Unit tests (optional).** New utility functions/classes (not depending on a
LAMMPS object) → add unit tests under `unittest/`. New force-computing styles
(pair, bond, …) with an existing tester → add a `.yaml` config + reference data.

**Citation reminder (optional).** You may register **one** citation — the single
most relevant paper *you/your group* authored on the feature — so LAMMPS prints a
reminder when the feature is used. **Do not** register third-party papers (e.g.
Nosé–Hoover for a thermostat) this way; those go in the doc page instead.

---

## 7. The GitHub workflow (what happens after you submit)

- **PRs are the only way anything enters LAMMPS** — even core developers submit
  PRs. Consider opening a **Draft PR** early to get CI feedback and non-binding
  design comments before it's "ready for review."
- On submit: CI runs (compile matrix, unit tests, doc HTML/PDF build, spell/label
  checks). Fix failures; CI re-runs on each push.
- A core developer self-assigns and does a technical review, applying labels.
  Two important ones: **`needs_work`** (the ball is in *your* court) and
  **`work_in_progress`** (a developer will make changes). Significant
  contributions may earn you a "collaborator" invitation.
- Merge requires: **all CI green + at least one developer approval**, and the
  merger differs from the approver, so **≥2 developers** see your code.
- Expect requested changes that "don't make sense to you" — they often matter to
  long-term maintenance. Ask questions freely; git's learning curve is understood.
- If you won't use GitHub, you can email a **gzipped tar** of new/changed files or
  a `diff -u`/`diff -c` patch (gzip only — no RAR/7-Zip) to the developers and ask
  them to open the PR. This is slower and depends on developer availability.

---

## 8. Quick pre-submission checklist

- [ ] Discussed with developers / opened an issue (for non-trivial work)
- [ ] Rebased on current `develop`, includes its latest bugfixes
- [ ] Compiles & runs correctly: SMALLBIG **and** BIGBIG, serial **and** MPI
- [ ] Lives in the right package (or a new `FOO` package with a `README`)
- [ ] GPL-2.0-only header + author name/email on every new file
- [ ] `.rst` docs in `doc/src/`, American English, ASCII-only; `make html` &
      `make spelling` clean; command tables/lists updated
- [ ] Names follow the lowercase/`compute_some_name`/`ComputeSomeName` convention
- [ ] `clang-format -i` run on your files; style-header macro wrapped & `;`-terminated
- [ ] Pointer members `nullptr`-initialized in ctor; no header member inits; no
      `using` in headers; forward declarations / PIMPL used
- [ ] `static constexpr` constants (no `#define`); no alternative tokens; rank-0 I/O
- [ ] Small fast example added (naming + no forbidden commands)
- [ ] Unit/YAML tests where applicable; single self-authored citation only
- [ ] CI green; ready to respond to `needs_work`

---

## Sources

- [3.2 Submitting new features for inclusion in LAMMPS](https://docs.lammps.org/Modify_contribute.html)
- [3.3 Requirements for contributions to LAMMPS](https://docs.lammps.org/Modify_requirements.html)
- [3.4 LAMMPS programming style](https://docs.lammps.org/Modify_style.html)
- [CONTRIBUTING.md (lammps/lammps, develop)](https://github.com/lammps/lammps/blob/develop/.github/CONTRIBUTING.md)
- [Contributing to LAMMPS (lammps.org)](https://www.lammps.org/contribute.html)
- [LAMMPS-GUI Development Guidelines (Axel Kohlmeyer)](https://lammps-gui.lammps.org/guidelines.html)
- [Kohlmeyer forum remark on open-source expectations](https://lammps.sandia.gov/threads/msg08148.html)
- [Berger, Kohlmeyer et al., "LAMMPS: A Case Study For Applying Modern Software Engineering…" (arXiv:2505.06877)](https://arxiv.org/abs/2505.06877)
