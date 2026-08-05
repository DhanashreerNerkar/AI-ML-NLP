---------------------------------------------------------------------------------------------------------------------
# PantryPilot (FoodPlanner.AI)
---------------------------------------------------------------------------------------------------------------------

## Quick start

```bash
cd FoodPlanner.AI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then set ANTHROPIC_API_KEY and SPOONACULAR_API_KEY
streamlit run app.py
```

FoodPlanner.AI is an **interactive chatbot** (not a static form). Flow:

Profile setup → planning questions → fridge photo / typed inventory → confirmation gate → freshness → diet-safe recipes → multi-day plan → meal feedback → substitutions / homemade derivation → confirmed shopping list.

Offline checks (no API keys required):

```bash
pytest -q
python scripts/smoke_demo.py
```

Profile persists in `data/user_data/user_profile.json`. Session memory resets on **New Plan**; the profile does not.

## Waste tracker (long-term memory)

Every **confirmed** inventory (photo or typed) is saved as a per-user snapshot in
`data/user_data/inventory_history.json`: each ingredient with its shelf-life category,
freshness score/status, and a summary (total items, at-risk count/%, top 3–4 use-first items).
Raw unconfirmed detections are never stored.

- **Duplicate photos:** uploads are hashed; re-uploading the same image within 48 h is flagged
  as a possible duplicate and the bot asks "same inventory or new groceries?" instead of
  silently counting a new purchase.
- **Outcomes, not assumptions:** if a previously at-risk item is missing from the next
  inventory, the bot asks what happened (Used / Still have it / Bought again / Spoiled /
  Thrown away / Donated / Not sure). A disappearance alone is recorded as *unresolved* —
  only user-confirmed "spoiled" or "thrown away" count as waste
  (`data/user_data/ingredient_outcomes.json`).
- **Tracer UI:** the sidebar **Waste tracker** button shows current & average at-risk %,
  the at-risk trend, repeatedly at-risk ingredients, unresolved items, confirmed waste, and
  conservative purchase suggestions (e.g., "buy ~25% less spinach"), each with its reason and
  confidence. Recommendations require ≥3 confirmed snapshots and the same ingredient at risk
  in ≥2 of them; suggested reductions are clamped to 10–50%. No prices or monetary estimates.
- **Chat memory:** new chats greet you with your recurring use-first ingredients and average
  at-risk percentage, without ever claiming something was wasted unless you confirmed it.

Stack: Streamlit chat UI · LangGraph orchestration · Claude (vision/planning) · Spoonacular (recipes) · deterministic diet/gap validators.

### Streamlit Community Cloud
1. Push this repo to GitHub.
2. Create an app at https://share.streamlit.io pointing at `app.py`.
3. Add secrets: `ANTHROPIC_API_KEY`, `SPOONACULAR_API_KEY` (optional `ANTHROPIC_MODEL`).
4. Demo mode works with fixture recipes if Spoonacular is unavailable.

---------------------------------------------------------------------------------------------------------------------
Purpose
Chabot will have profile of person having details about his dietry and traditiona/cultural eating habits/any other constraints.
A chatbot-based meal-planning assistant that accepts an uploaded or camera-captured fridge/pantry photo as its primary input.
Converts messy, real-world fridge/pantry contents into a multi-day meal plan as asked by user.
Optimizes for waste reduction, prioritizing ingredients closest to spoiling.
Uses that ingredients to search for recipe that can be prepared. 

Target user
Budget- and time-conscious home cooks (students, professionals, parents) planning meals on a weekly cycle.

Requirement 1 — Preference capture (happens first, before any plan is generated)
At session start, capture dietary type (vegetarian, vegan, non-vegetarian, etc.) and cultural/religious eating habits (e.g., halal, kosher, Jain).
Once a restriction is set, it is a hard constraint: no non-vegetarian dishes suggested to a vegetarian user, and no non-vegetarian ingredients allowed inside any suggested dish (including hidden ones like gelatin or fish sauce).

Requirement 2 — Ingredient substitution via home-makeable alternatives
If a required ingredient is missing, don't just suggest a store-bought swap — check whether the user can make it from other things likely at home.
Example: user wants palak paneer but has no paneer. If they have spinach (already flagged as aging) but no paneer, the bot asks whether they have milk or curd on hand.
Bot searches for the ways to make paneer/alternatives to panner. presents that to user and user chooses from given option.
If they select make paneer from milk steps, the bot walks them through making fresh paneer at home: boil milk 2–3 minutes, add an acid (vinegar or lemon juice), strain the curds.
This substitution logic should be a general pattern — recognizable "base ingredients → homemade derivative" chains, not a one-off for paneer keeping in context the users eating constraints

---------------------------------------------------------------------------------------------------------------------
FoodPlanner.AI — Requirements Folded into the Pipeline: 
---------------------------------------------------------------------------------------------------------------------
New Stage 0 — Profile Capture (runs before Stage 1, one per user): Triggered on first session or whenever profile is empty/update profile is clicked; captures diet type (vegetarian/vegan/non-veg) and cultural/religious constraints (halal, kosher, Jain, etc.).
Written to user_profile.json (long-term memory) as a hard constraint, not a preference — every later stage reads it, none can override it.
Also captures ambient rules implied by the diet type up front — e.g., Jain → no onion/garlic/root vegetables — so Stage 4/5 don't have to re-derive them.
Skipped on returning sessions unless the user asks to edit settings.

Stage 1 — Vision Ingestion (unchanged) : Extracts food items from the fridge/pantry photo → confirmed list via user gate.
Stage 2 — Freshness Scoring (unchanged) : Ranks confirmed items 5 (use now) → 1 (shelf-stable) using the shelf-life table.
Stage 3 — Critical-Priority Filter (unchanged, deterministic) : Pulls items scoring ≥4 into critical_priority.
Stage 4 — Meal Planning : Recipe search now filters Spoonacular candidates against the Stage 0 profile before they ever reach the LLM — no non-vegetarian dish or 
          hidden non-veg ingredient (gelatin, fish sauce, lard) can enter recipe_candidates. Plans are built to consume critical_priority items first, same as today.
Stage 5 — Substitution (this is where your new ask lives) : This stage gets restructured from "one compliant swap" into a two-step, profile-aware, interactive flow:
          Trigger: a planned recipe needs an ingredient the user doesn't have (e.g., paneer for palak paneer). Homemade-derivation search (new): before falling back to a 
          generic store-bought substitute, check a new derivation_kb — a table of "base ingredient(s) → homemade derivative" chains 
          (milk/curd → paneer; yeast+flour+time → sourdough starter; cream → butter, etc.). Availability check against inventory: ask/confirm whether the user has the base 
          ingredient(s) needed for the derivation (e.g., "Do you have milk or curd?"). Present options, not a decision: show the user a short list — e.g.,
          "Make paneer from milk," "Make paneer from curd," "Skip and use a different protein" — and let them pick. Diet-constraint filter applied to every option before 
          display: any derivation path or fallback substitute that conflicts with Stage 0's profile is removed from the list before the user ever sees it 
          (never offer a dairy-derived option to a vegan user, for instance). Guided steps on selection: once the user picks an option, the bot returns the step-by-step 
          process (e.g., boil milk 2–3 min → add vinegar/lemon → strain). Refusal (unchanged): if no compliant derivation and no compliant store-bought substitute exists, 
          return the fixed refusal object.

---------------------------------------------------------------------------------------------------------------------
Timeline: 1 Week · Stack: LangGraph + Claude + Streamlit + Spoonacular/USDA APIs (As prequired)
---------------------------------------------------------------------------------------------------------------------
P1 — Pipeline/Backend Development: LangGraph graph, stage prompts, orchestration logic
P2 — Data/Integration Development: KBs, external APIs (Spoonacular, USDA), profile/session schemas
P3 — Frontend/QA Development: Streamlit UI, testing harness, deployment

Stage map (what we're building)
→ Stage 0 Profile Capture 
→ Stage 1 Vision Ingestion (LLM) 
→ Stage 2 Freshness Scoring (LLM) 
→ Stage 3 Critical-Priority Filter (code) 
→ Stage 4 Meal Planning (LLM+RAG) 
→ Stage 5 Substitution/Derivation (LLM, interactive) 
→ Stage 6 Gap List (code)

--------------------------------
user-facing feature Covered
--------------------------------
1. Personalized Profile & Dietary Safety (T4, T7, T14)
Users set diet type and cultural/religious constraints (vegetarian, halal, kosher, Jain, etc.) once at the start. This becomes a hard constraint enforced by deterministic code — not LLM judgment — at every downstream stage, so a restriction can never be silently violated.

2. Fridge/Pantry Photo → Confirmed Inventory (T8)
Users upload or snap a photo of their fridge/pantry. Claude's vision extracts food items into an editable checklist, which the user must confirm before anything else runs — this stops vision misreads from cascading through the whole pipeline.

3. Spoilage-Aware Freshness Ranking (T9, T10)
Every confirmed ingredient gets scored 1–5 for perishability using a grounded shelf-life table (never guessed from the photo itself). Items scoring 4+ are flagged as "use first" — this is the core waste-reduction engine of the app.

4. Diet-Filtered Recipe Retrieval (T11, T12)
Live recipe search (Spoonacular) is filtered against the user's profile before any recipe reaches the planning model — so a vegetarian user's candidate list never contains meat, hidden gelatin, or fish sauce in the first place. Nutrition data (USDA) enriches results where available.

5. Multi-Day Meal Plan (T13)
A constrained-generation planner builds a multi-night plan using only the retrieved, filtered recipes — prioritizing soon-to-spoil ingredients first and reusing ingredients across nights to minimize waste.

6. Smart Ingredient Substitution — including "make it at home" (T15)
The standout feature: if a recipe needs something the user doesn't have (e.g., paneer), the bot first checks whether it can be derived at home from something the user likely has (milk/curd → paneer), presents those options interactively, and only falls back to "buy it" if no home-makeable or store-bought substitute is compliant with the user's diet.

7. Gap-Only Shopping List (T16)
Once a plan (and any substitution choices) are finalized, deterministic set-math computes exactly what's missing — no duplicated pantry staples, no LLM arithmetic errors.

8. Conversational Flow Control (T17, T18)
LangGraph routes each user message to the right stage (planning vs. substitution) and Streamlit renders the whole journey — photo in, plan and shopping list out — as a single interactive experience.

9. Safety & Reliability Hardening (T19, T20, T21)
Automated validators catch schema breaks and dietary violations before anything reaches the user; the pipeline is tested for consistency under reworded/typo'd input; and it's stress-tested against prompt injection, jailbreak attempts, and tool misuse.

10. Ethics & Responsible Rollout (T22, T23)
A documented check for how well the app serves halal/kosher/vegan/non-Western diets (an honest known-gap, not a fixed problem), plus a live, publicly deployable app on Streamlit Community Cloud.

------------------------------------------------------------------------------------------------------------------
Day 1:
------------------------------------------------------------------------------------------------------------------
T1 — Repo scaffolding: folder structure, requirements.txt, .env template, LangGraph + Claude SDK installed
Owner: P3
Depends on: none
Acceptance criteria: python app.py boots an empty Streamlit shell without error

T2 — Define JSON I/O contracts for every stage (detected_items, ranked, critical_priority, plan, substitution_result, gap_list)
Owner: P1
Depends on: none
Acceptance criteria: One schemas.py (or .json) file with a Pydantic/JSON-schema model per stage, reviewed by team

T3 — Build static KBs: shelf_life_kb.json, substitution_kb.json, derivation_kb.json (new — base-ingredient → homemade-derivative chains, diet-tagged)
Owner: P2
Depends on: none
Acceptance criteria: Each KB has >15 entries; every derivation entry has base_ingredients, method_steps, diet_tags

T4 — Define user_profile.json schema (diet_type, cultural/religious constraints, servings, dislikes) and session.json schema (confirmed items, freshness scores, pipeline state)
Owner: P2
Depends on: none
Acceptance criteria: Schemas cover Requirement 1 fields; sample fixture files committed

T5 — Spoonacular + USDA FoodData Central API keys obtained; thin wrapper functions stubbed (search_by_ingredients(), get_nutrition())
Owner: P2
Depends on: none
Acceptance criteria: Wrapper returns real API response for a hardcoded test query

T6 — Streamlit shell: image upload widget, chat panel placeholder, session-state scaffolding
Owner: P3
Depends on: none
Acceptance criteria: User can upload a photo and see it rendered; no backend logic yet

End of Day 1 checkpoint: schemas, KBs, API stubs, and UI shell all exist independently — nothing blocks Day 2.

------------------------------------------------------------------------------------------------------------------
DAY 2 — Core pipeline (depends on Day 1 outputs)
------------------------------------------------------------------------------------------------------------------

T7 — Stage 0: Profile capture flow — on first session, ask diet type + cultural/religious constraints, derive ambient rules (e.g., Jain → no onion/garlic), write to user_profile.json
Owner: P3
Depends on: T4, T6
Acceptance criteria: Fresh session always asks profile questions before any image upload is accepted

T8 — Stage 1: Vision ingestion prompt (Claude, image input) + user-confirmation gate in UI
Owner: P1
Depends on: T2, T6
Acceptance criteria: Uploading a fridge photo returns an editable checklist; nothing downstream fires until user confirms

T9 — Stage 2: Freshness scoring prompt, grounded in shelf_life_kb.json
Owner: P1
Depends on: T2, T3
Acceptance criteria: Confirmed list returns descending-sorted scores 1–5 with in_reference flag

T10 — Stage 3: Critical-priority filter (pure code, score >= 4)
Owner: P1
Depends on: T9
Acceptance criteria: Deterministic function, unit-tested, no LLM call

T11 — Spoonacular retrieval wired to real pipeline, filtered by Stage 0 profile before results reach any prompt (strip non-veg/hidden non-veg items)

Owner: P2
Depends on: T4, T5, T7
Acceptance criteria: A vegetarian profile never returns a candidate containing meat, gelatin, or fish sauce

T12 — USDA nutrition enrichment attached to recipe candidates (optional, non-blocking)
Owner: P2
Depends on: T5
Acceptance criteria: Nutrition fields populate when available; pipeline doesn't fail if USDA call errors

End of Day 2 checkpoint: a photo can go in and a filtered, freshness-ranked, profile-safe recipe candidate set comes out.
-----------------------------------------------------------------------------
DAY 3 — Planning, substitution, safety (depends on Day 2)
-------------------------------------------------------------------------------
T13 — Stage 4: Meal planning prompt — constrained generation, recipes drawn only from recipe_candidates, critical items scheduled first
Owner: P1
Depends on: T10, T11
Acceptance criteria: Plan JSON validates against schema; every ingredient traces to inventory or candidate list

T14 — Deterministic hard-constraint filter: cross-check every plan/substitution output against user_profile.json restrictions post-generation (belt-and-suspenders on top of T11)
Owner: P1
Depends on: T4, T13
Acceptance criteria: Any violating output is caught and blocked before reaching the UI — 0 violations in test set

T15 — Stage 5: Substitution/derivation flow — on missing ingredient → query derivation_kb.json → filter options by profile → present choices → return guided steps on selection; fixed refusal if nothing compliant
Owner: P2
Depends on: T3, T13
Acceptance criteria: Palak-paneer + no-paneer scenario surfaces "make from milk" / "make from curd" / "skip" — never a non-compliant option, e.g. never shown to a vegan

T16 — Stage 6: Gap-list (pure set-math: required − inventory − pantry staples), formatted markdown by LLM only
Owner: P1
Depends on: T13, T15
Acceptance criteria: If user picks "skip, buy paneer," paneer correctly appears in gap list

T17 — LangGraph wiring: all nodes connected, conditional router (intent = plan vs intent = substitute)
Owner: P1
Depends on: T13, T15, T16
Acceptance criteria: Graph runs end-to-end for both intents without manual stage-calling

T18 — Streamlit UI: render plan, gap list, substitution option buttons, refusal messages
Owner: P3
Depends on: T17
Acceptance criteria: Full user journey clickable in the browser, photo to shopping list

End of Day 3 checkpoint: full pipeline works end-to-end for both the happy path and the substitution path.
----------------------------------------------------------------------
DAY 3.5–4 — Hardening, testing, deploy (depends on full pipeline)
----------------------------------------------------------------------
T19 — Deterministic validators: JSON-schema check, banned-ingredient scan, gap-list math check, run before every user-facing response
Owner: P1
Depends on: T14, T16
Acceptance criteria: Validator suite runs in CI/script; 0 constraint violations across test set

T20 — Sensitivity/stability test script: synonym swaps, reordering, typos, reworded constraints, 5x-at-temp-0 repeatability
Owner: P3
Depends on: T17
Acceptance criteria: Score drift ≤1 point; structurally identical plans across reworded inputs

T21 — Self red-team pass: prompt injection ("ignore restrictions"), jailbreak/roleplay, tool misuse (mistitled recipe), data exposure attempt
Owner: P2
Depends on: T17
Acceptance criteria: All 4 attack classes blocked; results logged in a short report

T22 — Bias/ethics spot-check: same inventory run across halal, kosher, vegan, non-Western profiles
Owner: P2
Depends on: T11, T15
Acceptance criteria: Gaps in substitution/derivation quality documented (not necessarily fixed) as a known-limitation note

T23 — Deployment: Streamlit Cloud (or Docker) deploy + README + 2-minute demo script
Owner: P3
Depends on: T18, T19
Acceptance criteria: Public/shareable URL works end-to-end for a fresh user

Never cut: T14 and T19 — these are the safety backbone of the whole app.

------------------------------------------
Dependency chain at a glance
---------------------------------------------
<img width="1028" height="826" alt="image" src="https://github.com/user-attachments/assets/3827a7bf-5ce4-4269-83d2-75b337a3527f" />

------------------------------------------
Scope-cut list (if running out of time)
------------------------------------------
Cut in this order without breaking the core demo: T12 (USDA nutrition) → T22 (bias check, keep as documented gap) → T20 (full sensitivity matrix, keep 2–3 spot checks) → T21 (keep only prompt-injection + tool-misuse, drop the rest).

Never cut: T14 (hard-constraint filter) and T19 (validators) — these are the safety backbone of the whole app.

--------------------------------
Deply on: Streamlit Community Cloud - Free, built for exactly this, one-click from a GitHub repo
---------------------------------


